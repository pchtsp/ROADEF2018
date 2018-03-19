import package.instance as inst
import package.data_input as di


if __name__ == "__main__":
    # get_model_data()
    case_name = 'A3'
    model_data = di.get_model_data(case_name=case_name)
    instance = inst.Instance(model_data)
    batch = instance.get_batch()
    defects = instance.get_defects()

