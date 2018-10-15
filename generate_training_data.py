import sys


class GenerateTrainingData:

    def generate(self, count, seed):
        if seed>1 or seed<0:
            print("Please enter a seed between 0 and 1")
            return
        x = seed
        print("n,n+1")
        for i in range(0,count):
            y = x * 4.0 * (1.0 - x)
            print("{},{}".format(x,y))
            x = y


if __name__ == '__main__':
    if len(sys.argv)<3:
        print("Please pass the number of rows and a seed as arguments!")
    count = int(sys.argv[1])
    seed = float(sys.argv[2])
    gtd = GenerateTrainingData()
    gtd.generate(count, seed)