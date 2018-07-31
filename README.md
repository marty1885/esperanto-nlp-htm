# esperanto-nlp-htm

## Build and run
```
#generates the dataset for HTM to learn from
python genToken.py data.txt dataset.npy
python genToken.py test.txt test.npy
#compile the program
g++ main.cpp -o main -O3 -lnupic_core
#run the model
./main
```
