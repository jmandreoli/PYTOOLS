from cmd import Cmd

if __name__=='__main__':
    import sys
    sys.path.insert(0,'docstack.dir')
    from local import Manager
    mgr = Manager('docstack.dir',etype='SGE')
    
