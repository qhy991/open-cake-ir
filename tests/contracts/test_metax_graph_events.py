"""Compiled graph submission with deterministic CPU APIs and exact work accounting."""
import ctypes as C
import unittest
from open_cake_ir.evaluation import metax_native_events as native


class GraphSubmissionTests(unittest.TestCase):
    def test_each_graph_owns_one_reset_one_target_and_a_fresh_output(self):
        Create=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_void_p),C.c_uint)
        Record=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p)
        RecordFlags=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p,C.c_uint)
        Event=C.CFUNCTYPE(C.c_int,C.c_void_p)
        Elapsed=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_float),C.c_void_p,C.c_void_p)
        Reset=C.CFUNCTYPE(C.c_int,C.c_size_t,C.c_uint,C.c_size_t,C.c_void_p)
        Launch=C.CFUNCTYPE(C.c_int,C.c_void_p,*([C.c_uint]*7),C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p)
        Begin=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_int)
        End=C.CFUNCTYPE(C.c_int,C.c_void_p,C.POINTER(C.c_void_p))
        Instantiate=C.CFUNCTYPE(C.c_int,C.POINTER(C.c_void_p),C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t)
        Nodes=C.CFUNCTYPE(C.c_int,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t))
        NodeType=C.CFUNCTYPE(C.c_int,C.c_void_p,C.POINTER(C.c_int))
        count=[0];current=[None];graphs={};actual=[];closed=[];flags=[]
        def create(out,flag):count[0]+=1;out[0]=count[0];return 0
        def create_stream(out,flag):out[0]=9;flags.append(flag);return 0
        def record(event,stream):
            if current[0] is not None:current[0].append(('event',event,stream))
            return 0
        def reset(address,value,words,stream):
            current[0].append(('reset',address,value,words,stream));return 0
        def launch(function,gx,gy,gz,bx,by,bz,shared,stream,args,extra):
            current[0].append(('target',C.cast(args[0],C.POINTER(C.c_void_p)).contents.value,stream));return 0
        def begin(stream,mode):current[0]=[];return 0
        def end(stream,out):
            number=len(graphs)+100;graphs[number]=current[0];current[0]=None;out[0]=number;return 0
        def instantiate(out,graph,error,log,size):out[0]=graph;return 0
        def graph_launch(graph,stream):actual.append(graphs[graph]);return 0
        def elapsed(out,begin,end):out[0]=.001;return 0
        def close(value):closed.append(value);return 0
        event_flags=[]
        def record_flags(event,stream,flag):event_flags.append(flag);return record(event,stream)
        def nodes(graph,out,count):
            count[0]=4
            if out:
                for i in range(4):out[i]=graph*10+i
            return 0
        def node_type(node,out):out[0]=[2,7,0,7][node%10];return 0
        callbacks=[Create(create),Record(record),Event(lambda event:0),Elapsed(elapsed),Event(close),
            Reset(reset),Launch(launch),Event(lambda stream:0),Create(create_stream),Event(close),
            Begin(begin),End(end),Instantiate(instantiate),Record(graph_launch),Event(close),Event(close),
            RecordFlags(record_flags),Nodes(nodes),NodeType(node_type)]
        api=(C.c_void_p*19)(*(C.cast(callback,C.c_void_p).value for callback in callbacks))
        values=[C.c_void_p(i+1000) for i in range(16)]
        slots=[(C.c_void_p*1)(C.addressof(value)) for value in values]
        pointers=(C.POINTER(C.c_void_p)*16)(*slots)
        output,calls,phase=(C.c_float*5)(),C.c_uint(),C.c_uint()
        status=native.prepare_helper().cake_maca_graph_event_cohort(api,55,(C.c_uint*6)(1,1,1,64,1,1),
            0,pointers,11,5,1234,8388608,output,C.byref(calls),C.byref(phase))
        self.assertEqual((status,calls.value),(0,16));self.assertEqual(flags,[1])
        self.assertEqual(event_flags,[1]*32)
        self.assertEqual(len(graphs),16);self.assertEqual(len(actual),16)
        for i,graph in enumerate(actual):
            self.assertEqual(graph,[('reset',1234,0x3f800000,8388608,9),('event',1,9),
                                    ('target',1000+i,9),('event',2,9)])
        self.assertEqual(len(closed),35)  # Sixteen graph+exec pairs, two events, one stream.
        self.assertTrue(all(sample>0 for sample in output))
