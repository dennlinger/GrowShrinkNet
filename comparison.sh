
### CNN
## MNIST
# batch size
python3 main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 64
python3 main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 128

# filters
python3 main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 64
python3 main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 128

# kernel size
python3 main.py --data vMNIST --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 64
python3 main.py --data vMNIST --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 128

## Fashion-MNIST
# batch size
python3 main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 64
python3 main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 128

# filters
python3 main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 64
python3 main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 128

# kernel size
python3 main.py --data vFashion --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 64
python3 main.py --data vFashion --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 128


### gCNN
## MNIST
# batch size
python3 main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 64
python3 main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 128

# filters
python3 main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 64
python3 main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 128

# kernel size
python3 main.py --data vMNIST --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 64
python3 main.py --data vMNIST --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 128

## Fashion-MNIST
# batch size
python3 main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 64
python3 main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 128

# filters
python3 main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 64
python3 main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 128

# kernel size
python3 main.py --data vFashion --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 64
python3 main.py --data vFashion --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 128


### sCNN
## MNIST
# batch size
python3 main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 64
python3 main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 128

# filters
python3 main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 64
python3 main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 128

# kernel size
python3 main.py --data vMNIST --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 64
python3 main.py --data vMNIST --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 128

## Fashion-MNIST
# batch size
python3 main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 64
python3 main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 128

# filters
python3 main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 64
python3 main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 128

# kernel size
python3 main.py --data vFashion --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 64
python3 main.py --data vFashion --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 128


