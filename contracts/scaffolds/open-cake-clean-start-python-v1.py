from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="candidate", target="sm_100a", backend="triton", entry_point="cake_flash_kmeans_assign")
def candidate(lm, tokens: cake.Tensor((32, 65536, 128), "bf16", mode="input"), centroids: cake.Tensor((32, 1024, 128), "bf16", mode="input"), centroid_sq: cake.Tensor((32, 1024), "fp32", mode="input"), assignments: cake.Tensor((32, 65536), "int32", mode="output")):
    ...
