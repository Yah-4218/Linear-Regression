import LinRegV2
import LinRegV3
import numpy as np
import time

def main():

    V2_times = []
    V3_times = []

    ARRAY_SIZE = 15
    xAxis = np.array([i for i in range(ARRAY_SIZE)])
    yAxis = np.array([])
    for i in range(ARRAY_SIZE):
        value = np.array(20*xAxis[i]+2)
        yAxis = np.append(yAxis, value)


    for _ in range(10):
        V2_start = time.time()
        V2_model = LinRegV2.Linear_Regression(xAxis, yAxis, epochs = 10_000, learningRate = 0.001)
        V2_model.TrainModel()
        V2_end = time.time()

        V2_times.append(V2_end-V2_start)

    for _ in range(10):
        V3_start = time.time()
        V3_model = LinRegV3.Linear_Regression(xAxis, yAxis, epochs = 10_000, learningRate = 0.001)
        V3_model.TrainModel()
        V3_end = time.time()

        V3_times.append(V3_end-V3_start)

    print(f"V2 Times: {sum(V2_times)/len(V2_times)}")
    print(V2_times)
    print(f"V3 Times: {sum(V3_times)/len(V3_times)}")
    print(V3_times)



if __name__ == "__main__":
    main()


# Average Time for LinRegV2: 0.08391974449157714 - 10,000 epochs with 100 iterations (Array Size: 8)
# Average Time for LinRegV3: 0.10756600379943848 - 10,000 epochs with 100 iterations (Array Size: 8)

# Vectorisation thrives with large datasets. However, if I increase the Array Size to 100, then the data becomes too large, causing overflow errors.