from open_cake_ir.compiler import frontend as cake

@cake.schedule(name='xcore1002-fp8-resident-compensated-64', target='xcore1002',
               backend='triton', entry_point='cake_fp8_resident_compensated_64')
def matrix(lm, a: cake.Tensor((64, 64), 'fp8_e4m3'),
           b: cake.Tensor((64, 64), 'fp8_e4m3'),
           out: cake.Tensor((64, 64), 'fp32', mode='output')):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(a, axis=0, dimension=0, tile=2)
    with compute:
        left = lm.load(a[row, :], id='load_a')
        right = lm.load(b[:, :], id='load_b')
        result = lm.mma(left, right,
                        instruction={'contract': 'maca.simt.fp8e4m3_compensated_fp32'},
                        tile_shape=(2, 64, 64), id='compensated_mma')
        lm.store(out[row, :], result, id='store_out')
