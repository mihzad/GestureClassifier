import os
import asyncio
import concurrent.futures

import onnxruntime as ort
import numpy as np



class Inferencer():
    def __init__(self):

        self.modelname = os.getenv("ONNX_MODEL_FILE")
        self.prediction_threshold = float(os.getenv("ONNX_MODEL_PREDICTION_THRESHOLD"))
        self.class_names = os.getenv("ONNX_MODEL_CLASS_NAMES").split(",")
        self.class_names_noclass_label = os.getenv("ONNX_MODEL_CLASS_NAMES_NOCLASS")

        self.ortinf = ort.InferenceSession(self.modelname)

        
    async def run_model(self, numpy_tensor: np.ndarray) -> list[str]:

        concur_future = concurrent.futures.Future()

        def callback(outputs, user_data, err):
            if err:
                user_data.set_exception(RuntimeError(err))
            else:
                user_data.set_result(outputs[0])
        #print("starting run...")
        self.ortinf.run_async(
            output_names=[self.ortinf.get_outputs()[0].name],
             input_feed= {self.ortinf.get_inputs()[0].name: numpy_tensor},
             callback=callback,
             user_data=concur_future,
            )

        result = await asyncio.wrap_future(concur_future)
        o = np.array(result)
        #print("tonp:", o.shape)

        predictions_list = o.argmax(1)
        predicted_vals = o.max(1)
        # print("pred list", predictions_list.shape)
        # print("pred vals", predicted_vals.shape)
        # print(predictions_list)
        # print(predicted_vals)
        # print([np.sum(o[i]) for i in range(len(o))])
        # print(o)

        final_predictions_batch = []
        for i in range(len(predictions_list)):
            if predicted_vals[i] >= self.prediction_threshold:
                label = predictions_list[i]
                final_predictions_batch.append(self.class_names[label])
            else:
                final_predictions_batch.append(self.class_names_noclass_label)

        #print(final_predictions_batch)
        return final_predictions_batch



if __name__== "__main__": #simple testing
    example_inputs = np.load("np_examplevid_label6.npy")
    example_inputs = example_inputs[np.newaxis, :]


    example_inputs2 = np.random.rand(1,3,16,224,224).astype(np.float32)
    i = Inferencer()
    result = asyncio.run(i.run_model(example_inputs))
    print(result)
