import Darwin
import Foundation
import Metal

private let observationKind = "open_cake_metal_hierarchical_reduction_dispatch_v1"
private let entryPoint = "hierarchical_reduction_w32_probe"
private let specializedSIMDWidth = 32
private let simdgroupCounts = [1, 2, 4, 8, 16, 32]
private let scratchBytes = 144
private let epochDirtySentinels: [Float] = [-12345.25, 9876.5]

private enum ProbeError: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message): return message
        }
    }
}

private func require<T>(_ value: T?, _ message: String) throws -> T {
    guard let value else {
        throw ProbeError.failed(message)
    }
    return value
}

private func makeInputBuffer(
    _ device: MTLDevice,
    values: [Float],
    label: String
) throws -> MTLBuffer {
    guard !values.isEmpty else {
        throw ProbeError.failed("\(label) must not be empty")
    }
    let buffer = try values.withUnsafeBytes { bytes in
        try require(
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: bytes.count,
                options: .storageModeShared
            ),
            "failed to allocate \(label)"
        )
    }
    buffer.label = label
    return buffer
}

private func makeOutputBuffer(
    _ device: MTLDevice,
    count: Int,
    label: String
) throws -> MTLBuffer {
    let byteCount = count * MemoryLayout<Float>.stride
    let buffer = try require(
        device.makeBuffer(length: byteCount, options: .storageModeShared),
        "failed to allocate \(label)"
    )
    buffer.label = label
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    for index in 0..<count {
        pointer[index] = .nan
    }
    return buffer
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

private func epoch0Values(threadCount: Int) -> [Float] {
    (0..<threadCount).map { index in
        Float((index % 13) - 6) * 0.125 + 0.5
    }
}

private func epoch1Values(threadCount: Int) -> [Float] {
    (0..<threadCount).map { index in
        Float((index % 11) - 5) * 0.25 - 0.75
    }
}

private func fp32Sum(_ values: [Float]) -> Float {
    values.reduce(Float(0)) { partial, value in partial + value }
}

private func commandStatus(_ status: MTLCommandBufferStatus) -> String {
    switch status {
    case .notEnqueued: return "not_enqueued"
    case .enqueued: return "enqueued"
    case .committed: return "committed"
    case .scheduled: return "scheduled"
    case .completed: return "completed"
    case .error: return "error"
    @unknown default: return "unknown"
    }
}

private func validateExact(
    _ value: Float,
    expected: Float,
    context: String
) throws -> Float {
    guard value.isFinite else {
        throw ProbeError.failed("\(context) was not written with a finite value")
    }
    guard value.bitPattern == expected.bitPattern else {
        throw ProbeError.failed(
            "\(context) differs from the exact FP32 oracle: expected bits "
                + "\(expected.bitPattern), observed bits \(value.bitPattern)"
        )
    }
    return abs(value - expected)
}

private func runCase(
    device: MTLDevice,
    queue: MTLCommandQueue,
    pipeline: MTLComputePipelineState,
    simdgroupCount: Int
) throws -> [String: Any] {
    let threadCount = simdgroupCount * specializedSIMDWidth
    guard threadCount <= pipeline.maxTotalThreadsPerThreadgroup else {
        throw ProbeError.failed(
            "\(threadCount)-thread case exceeds pipeline maxTotalThreadsPerThreadgroup"
        )
    }

    let inputs = [
        epoch0Values(threadCount: threadCount),
        epoch1Values(threadCount: threadCount),
    ]
    let expected = inputs.map(fp32Sum)
    let inputBuffers = try inputs.enumerated().map { epoch, values in
        try makeInputBuffer(device, values: values, label: "epoch\(epoch)_input")
    }
    let broadcastBuffer = try makeOutputBuffer(
        device,
        count: 2 * threadCount,
        label: "broadcast_outputs"
    )
    let ownerBuffer = try makeOutputBuffer(device, count: 2, label: "owner_results")
    let snapshotBuffer = try makeOutputBuffer(device, count: 4, label: "dirty_snapshots")

    let commandBuffer = try require(queue.makeCommandBuffer(), "failed to create command buffer")
    let encoder = try require(
        commandBuffer.makeComputeCommandEncoder(),
        "failed to create compute encoder"
    )
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffers[0], offset: 0, index: 0)
    encoder.setBuffer(inputBuffers[1], offset: 0, index: 1)
    encoder.setBuffer(broadcastBuffer, offset: 0, index: 2)
    encoder.setBuffer(ownerBuffer, offset: 0, index: 3)
    encoder.setBuffer(snapshotBuffer, offset: 0, index: 4)
    encoder.setThreadgroupMemoryLength(scratchBytes, index: 0)
    encoder.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        throw ProbeError.failed("\(threadCount)-thread command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.failed(
            "\(threadCount)-thread command buffer status was "
                + "\(commandStatus(commandBuffer.status))"
        )
    }

    let broadcasts = readFloats(broadcastBuffer, count: 2 * threadCount)
    let ownerResults = readFloats(ownerBuffer, count: 2)
    let dirtySnapshots = readFloats(snapshotBuffer, count: 4)
    var epochResults = [[String: Any]]()

    for epoch in 0..<2 {
        let ownerError = try validateExact(
            ownerResults[epoch],
            expected: expected[epoch],
            context: "case \(simdgroupCount), epoch \(epoch) owner result"
        )
        let snapshotOffset = epoch * 2
        let partialSentinelError = try validateExact(
            dirtySnapshots[snapshotOffset],
            expected: epochDirtySentinels[epoch],
            context: "case \(simdgroupCount), epoch \(epoch) dirty partial snapshot"
        )
        let scalarSentinelError = try validateExact(
            dirtySnapshots[snapshotOffset + 1],
            expected: epochDirtySentinels[epoch],
            context: "case \(simdgroupCount), epoch \(epoch) dirty scalar snapshot"
        )

        let epochBroadcasts = broadcasts[(epoch * threadCount)..<((epoch + 1) * threadCount)]
        var maxBroadcastError = Float(0)
        var mismatchCount = 0
        for (thread, observed) in epochBroadcasts.enumerated() {
            do {
                let error = try validateExact(
                    observed,
                    expected: expected[epoch],
                    context: "case \(simdgroupCount), epoch \(epoch), thread \(thread) broadcast"
                )
                maxBroadcastError = max(maxBroadcastError, error)
            } catch {
                mismatchCount += 1
            }
        }
        guard mismatchCount == 0 else {
            throw ProbeError.failed(
                "case \(simdgroupCount), epoch \(epoch) has \(mismatchCount) "
                    + "broadcast mismatches"
            )
        }

        epochResults.append([
            "epoch": epoch,
            "input_pattern": epoch == 0
                ? "((thread_mod_13)-6)*0.125+0.5"
                : "((thread_mod_11)-5)*0.25-0.75",
            "dirty_sentinel": epochDirtySentinels[epoch],
            "observed_dirty_partial_slot": dirtySnapshots[snapshotOffset],
            "observed_dirty_scalar_slot": dirtySnapshots[snapshotOffset + 1],
            "dirty_snapshot_max_abs_error": max(partialSentinelError, scalarSentinelError),
            "dirty_snapshot_mismatch_count": 0,
            "dirty_sentinel_preserved": true,
            "expected_sum": expected[epoch],
            "observed_owner_sum": ownerResults[epoch],
            "owner_abs_error": ownerError,
            "owner_mismatch_count": 0,
            "broadcast_thread_count": threadCount,
            "broadcast_mismatch_count": mismatchCount,
            "broadcast_max_abs_error": maxBroadcastError,
            "mismatch_count": 0,
            "passed": true,
        ])
    }

    return [
        "group_count": simdgroupCount,
        "threads_per_threadgroup": threadCount,
        "threadgroup_memory_bytes": scratchBytes,
        "command_buffer_status": commandStatus(commandBuffer.status),
        "epochs": epochResults,
        "passed": true,
    ]
}

private func runProbe(metallibPath: String) throws -> [String: Any] {
    guard metallibPath.hasPrefix("/") else {
        throw ProbeError.failed("metallib path must be absolute")
    }
    let metallibURL = URL(fileURLWithPath: metallibPath)
    let metallibBytes = try Data(contentsOf: metallibURL, options: .mappedIfSafe)
    guard metallibBytes.count >= 4,
          Array(metallibBytes.prefix(4)) == [0x4d, 0x54, 0x4c, 0x42] else {
        throw ProbeError.failed("metallib is missing the MTLB header")
    }

    let device = try require(MTLCreateSystemDefaultDevice(), "Metal device is unavailable")
    guard device.name == "Apple M4" else {
        throw ProbeError.failed("probe requires the allowlisted Apple M4 device class")
    }
    let supportsApple9OrNewer = device.supportsFamily(.apple9)
    guard supportsApple9OrNewer else {
        throw ProbeError.failed("Metal device does not support Apple GPU family 9 or newer")
    }
    let library = try device.makeLibrary(URL: metallibURL)
    let function = try require(
        library.makeFunction(name: entryPoint),
        "metallib entry point \(entryPoint) is missing"
    )
    let pipeline = try device.makeComputePipelineState(function: function)
    guard pipeline.threadExecutionWidth == specializedSIMDWidth else {
        throw ProbeError.failed(
            "width-32 specialization refused pipeline width \(pipeline.threadExecutionWidth)"
        )
    }
    guard pipeline.maxTotalThreadsPerThreadgroup >= 32 * specializedSIMDWidth else {
        throw ProbeError.failed(
            "pipeline cannot dispatch the required 32-SIMDgroup case"
        )
    }
    guard pipeline.staticThreadgroupMemoryLength + scratchBytes
            <= device.maxThreadgroupMemoryLength else {
        throw ProbeError.failed("pipeline static plus dynamic threadgroup memory exceeds device limit")
    }
    let queue = try require(device.makeCommandQueue(), "failed to create command queue")
    let cases = try simdgroupCounts.map { count in
        try runCase(device: device, queue: queue, pipeline: pipeline, simdgroupCount: count)
    }

    return [
        "schema_version": 1,
        "kind": observationKind,
        "status": "passed",
        "performance_measured": false,
        "scientific_claim_authorized": false,
        "probe_scope": "shared_choreography_and_rms_single_owner_broadcast_only",
        "non_claims": [
            "no_timing_or_performance_claim",
            "no_layer_norm_all_simdgroup_final_reducer_claim",
            "no_bfloat_input_conversion_claim",
            "no_compiler_lowering_or_ir_admission_claim",
        ],
        "device": [
            // A generic chip class is retained; no registry ID, UUID, serial,
            // hostname, or other stable device identifier is read or emitted.
            "device_class": "Apple M4",
            "supports_apple9_or_newer": supportsApple9OrNewer,
            "max_threadgroup_memory_bytes": device.maxThreadgroupMemoryLength,
        ],
        "pipeline": [
            "entry_point": entryPoint,
            "thread_execution_width": pipeline.threadExecutionWidth,
            "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup,
            "static_threadgroup_memory_bytes": pipeline.staticThreadgroupMemoryLength,
            "dynamic_threadgroup_memory_bytes": scratchBytes,
        ],
        "kernel_calls": cases.count,
        "coverage": [
            "simdgroup_counts": simdgroupCounts,
            "epochs_per_dispatch": 2,
            "single_owner_final_scalar_broadcast": true,
            "scratch_reused_within_dispatch": true,
            "uniform_reuse_barrier": "threadgroup_barrier(mem_threadgroup)",
            "layer_all_simdgroups_tested": false,
        ],
        "oracle": [
            "dtype": "float32",
            "tolerance_rule": "exact_fp32_bits_for_dyadic_fixture",
        ],
        "cases": cases,
    ]
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw ProbeError.failed("usage: run_hierarchical_reduction METALLIB")
    }
    let observation = try runProbe(metallibPath: CommandLine.arguments[1])
    guard JSONSerialization.isValidJSONObject(observation) else {
        throw ProbeError.failed("hierarchical-reduction observation is not valid JSON")
    }
    let encoded = try JSONSerialization.data(
        withJSONObject: observation,
        options: [.sortedKeys]
    )
    FileHandle.standardOutput.write(encoded)
    FileHandle.standardOutput.write(Data([0x0a]))
} catch {
    let message = "metal hierarchical reduction probe failed: \(error)\n"
    FileHandle.standardError.write(message.data(using: .utf8)!)
    exit(1)
}
