from space_optimizer.default_space import DefaultSpace

class InitSpace(DefaultSpace):
    def __init__(self, dbms, test, timeout, target_knobs_path, seed):
        super().__init__(dbms, test, timeout, target_knobs_path, seed)
        self.define_search_space()

    def define_search_space(self):
        for t_knob in self.target_knobs[:]:
            info = self.dbms.knob_info[t_knob]
            if info is None or info.get("vartype") is None:
                self.target_knobs.remove(t_knob)# this knob is not by the DBMS under specific version
                continue
            t_knob = self.get_default_space(t_knob, info)
            if t_knob is not None:
                self.search_space.add_hyperparameter(t_knob)
        