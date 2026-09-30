# SPDX-License-Identifier: Apache-2.0
"""Single-card Host length / device metadata A→B→C→A replay regression."""


def check_metadata_replay(config, layer, fixture, weights, call, args, report):
    import torch

    from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, restore
    from dsv4_csa_validation import compare_tensor
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.host_metadata import (
        CSAHostMetadata, validate_graph_replay,
    )
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.native_adapter import HBGNativeCSACall

    if args.history != 8192:
        raise ValueError("Host metadata boundary regression uses history=8192")
    result = {"status": "RUNNING", "scope": "同址更新长度、位置、页表、compact metadata；跨拓扑拒绝旧图并重新捕获",
              "steps": [], "cross_topology_rejected": False}
    report["hbg_metadata_replay"] = result
    fixtures = {"A": fixture}
    calls = {"A": call}
    for label, history in (("B", 8196), ("C", 8184)):
        other = make_fixture(config, layer.self_attn, args.batch, history, args.seed + history,
                             fixture["hidden"].device, table_history=8200, reverse_pages=True)
        groups = {name: (other["metadata"][g["prefix"]], tuple(g["views"]))
                  for name, g in other["groups"].items()}
        calls[label] = HBGNativeCSACall(
            call.ops, weights, other["hidden"], other["positions"], groups,
            layer_name=other["groups"]["compressed"]["prefix"], compact_metadata=other["compact"],
            host_metadata=CSAHostMetadata.from_native(groups["indexer"][0].decode),
        )
        fixtures[label] = other

    # Snapshot only actual root inputs; weights are shared and immutable.
    outputs = {"idx_topk_scores", "idx_topk", "x_out"}
    snapshots = {}
    reference = {}
    for label, source in calls.items():
        restore(fixtures[label])
        snapshots[label] = {name: value.detach().cpu().clone() for name, value in source.args.items()
                            if isinstance(value, torch.Tensor) and name not in weights and name not in outputs}
        for name, value in snapshots[label].items():
            if value.shape != call.args[name].shape or value.dtype != call.args[name].dtype:
                raise ValueError(f"Same-address test has incompatible {label}/{name} shape or dtype")
        source()
        torch.npu.synchronize()
        reference[label] = collect_state(fixtures[label], source.args["x_out"], source.args["idx_topk"])

    def load(label):
        source_fixture = fixtures[label]
        for name, group in fixture["groups"].items():
            group["allocation"].copy_(source_fixture["groups"][name]["initial"])
        for name, value in snapshots[label].items():
            call.args[name].copy_(value)
        call.update_host_metadata(hosts[label])

    def check(label):
        actual = collect_state(fixture, call.args["x_out"], call.args["idx_topk"])
        checks = {name: compare_tensor(value, reference[label][name], 0, 0) for name, value in actual.items()}
        source = fixtures[label]
        guards = guard_checks({"groups": {
            name: {**group, "initial": source["groups"][name]["initial"],
                   "allowed": source["groups"][name]["allowed"]}
            for name, group in fixture["groups"].items()
        }})
        # Root inputs include metadata views; every read-only one must survive.
        mutable = {"compress_state", "inner_compress_state", "kv_cache", "cmp_kv", "idx_native_kv_cache"}
        readonly = {name: compare_tensor(call.args[name], value, 0, 0)
                    for name, value in snapshots[label].items() if name not in mutable}
        if any(v["status"] != "PASS" for v in (*checks.values(), *guards.values(), *readonly.values())):
            raise AssertionError(f"HBG metadata replay mismatch for {label}: {checks}, {guards}, {readonly}")
        return {"comparison": checks, "guards": guards, "readonly": readonly}

    prefix = fixture["groups"]["indexer"]["prefix"]
    graphs = {}
    hosts = {label: source.host_metadata for label, source in calls.items()}
    for label in ("A", "B", "C", "A"):
        load(label)
        # The real ACLGraphWrapper uses this same validation before replay.
        current = fixtures[label]["metadata"]
        if label == "C":
            try:
                validate_graph_replay({prefix: hosts["A"]}, current)
            except ValueError as exc:
                if "changed Score/Top-K topology" not in str(exc):
                    raise
                result["cross_topology_rejected"] = True
            else:
                raise AssertionError("Cross-topology replay was not rejected")
        key = hosts[label].graph_key()
        if key not in graphs:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                call()
            graphs[key] = graph, hosts[label]
        graph, captured_host = graphs[key]
        validate_graph_replay({prefix: captured_host}, current)
        load(label)
        graph.replay()
        torch.npu.synchronize()
        result["steps"].append({"input": label, "host_max_seq_len": hosts[label].max_seq_len,
                                "graph_key": key, **check(label)})
    result.update(status="PASS", captures=len(graphs))

    # Exercise the actual service custom op as well: Host values must come
    # from Native metadata, rather than a scalar supplied by the test adapter.
    import os

    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.service import CSAServiceRuntime
    import vllm_ascend.ops.dsv4_csa  # noqa: F401

    os.environ["PTO_CSA_RUNTIME"] = "host_build_graph"
    wrapper = layer.self_attn.dsa_attn
    wrapper._pto_csa_layer = (layer,)
    runtime = CSAServiceRuntime(layer.self_attn, call.ops, 40, layer)
    wrapper._pto_csa_runtime = runtime
    for name, group in fixture["groups"].items():
        group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
    output = torch.empty_like(fixture["hidden"])

    def service_call():
        with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                       num_actual_tokens=fixture["tokens"]):
            context = get_forward_context()
            if not runtime.eligible(context, fixture["hidden"], fixture["positions"]):
                raise AssertionError("HBG service was not selected; Native fallback is not a passing result")
            torch.ops.vllm.dsv4_csa_forward(fixture["hidden"], fixture["positions"], output, wrapper.prefix)
            host_values = context.additional_kwargs.get("pto_csa_hbg_graph_metadata")
            if host_values != {prefix: hosts["A"]}:
                raise AssertionError(f"Service did not publish the captured Host contract: {host_values}")

    service_results = []
    load("A")
    service_call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        service_call()
    for run in (service_call, graph.replay):
        load("A")
        run()
        torch.npu.synchronize()
        actual = collect_state(fixture, output, runtime.topk[:fixture["tokens"]])
        checks = {name: compare_tensor(value, reference["A"][name], 0, 0) for name, value in actual.items()}
        guards = guard_checks(fixture)
        if any(v["status"] != "PASS" for v in (*checks.values(), *guards.values())):
            raise AssertionError(f"HBG service differs from the direct entry: {checks}, {guards}")
        service_results.append({"comparison": checks, "guards": guards})
    result["service"] = {"status": "PASS", "eager_then_graph": service_results}
