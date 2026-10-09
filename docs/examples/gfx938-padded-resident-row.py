from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="padded-resident-existing",target="gfx938",backend="triton",entry_point="run")
def candidate(lm,x:cake.Tensor((17,96),"fp32"),y:cake.Tensor((17,),"fp32",mode="output")):
    row=lm.program(x,axis=0,dimension=0,tile=1)
    column=lm.program(x,axis=1,dimension=1,tile=128)
    compute=lm.role(execution_groups=[0,1,2,3])
    with compute:
        value=lm.load(x[row,column])
        total=lm.reduce(value,op="sum",axis=0,scope="cta")
        lm.store(y[row],total,coalesced=False)
