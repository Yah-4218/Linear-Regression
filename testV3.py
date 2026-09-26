from LinRegV2 import *
from LinRegV3 import *
import time

def main():

    times = []

    for i in range(100):
        start = time.time()

        ARRAY_SIZE = 8
        xAxis = np.array([i for i in range(ARRAY_SIZE)])
        yAxis = np.array([])
        for i in range(ARRAY_SIZE):
            value = np.array(202*xAxis[i])
            yAxis = np.append(yAxis, value)

        model = Linear_Regression(xAxis, yAxis, epochs=10_000, learningRate=0.001)
        model.TrainModel()


        end = time.time()
        times.append(end-start)

    print(f"Times: {sum(times)/len(times)}")


if __name__ == "__main__":
    main()


# Average Time for LinRegV2: 0.1302497673034668 - 10,000 epochs with 100 iterations

# Average Time for LinRegV3: 