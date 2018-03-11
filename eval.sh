
### CNN
## MNIST
# batch size
python main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 64 --validate True
python main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 128 --validate True

# kernel size
python main.py --data vMNIST --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 128 --validate True

## Fashion-MNIST
# batch size
python main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 64 --validate True
python main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model CNN --gpu 1 --kernel 3 --num_filters 16 --batch_size 128 --validate True

# kernel size
python main.py --data vFashion --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model CNN --gpu 1 --kernel 5 --num_filters 16 --batch_size 128 --validate True


### gCNN
## MNIST
# batch size
python main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 64 --validate True
python main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 128 --validate True

# kernel size
python main.py --data vMNIST --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 128 --validate True

## Fashion-MNIST
# batch size
python main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 64 --validate True
python main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model gCNN --gpu 1 --kernelg 3 --num_filters 16 --batch_size 128 --validate True

# kernel size
python main.py --data vFashion --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model gCNN --gpu 1 --kernelg 5 --num_filters 16 --batch_size 128 --validate True


### sCNN
## MNIST
# batch size
python main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 64 --validate True
python main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 128 --validate True

# kernel size
python main.py --data vMNIST --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 128 --validate True

## Fashion-MNIST
# batch size
python main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 64 --validate True
python main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model sCNN --gpu 1 --kernels 7 --num_filters 16 --batch_size 128 --validate True

# kernel size
python main.py --data vFashion --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model sCNN --gpu 1 --kernels 9 --num_filters 16 --batch_size 128 --validate True


### gsCNN
## MNIST
# batch size
python main.py --data vMNIST --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 8 --batch_size 64 --validate True
python main.py --data vMNIST --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vMNIST --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 16 --batch_size 128 --validate True


## Fashion-MNIST
# batch size
python main.py --data vFashion --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 8 --batch_size 64 --validate True
python main.py --data vFashion --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vFashion --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model gsCNN --gpu 1 --kernels 7 --kernelg 3 --num_filters 16 --batch_size 128 --validate True

### gmsCNN
## MNIST
# batch size
python main.py --data vMNIST --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 8 --batch_size 64 --validate True
python main.py --data vMNIST --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vMNIST --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 16 --batch_size 64 --validate True
python main.py --data vMNIST --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 16 --batch_size 128 --validate True


## Fashion-MNIST
# batch size
python main.py --data vFashion --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 8 --batch_size 64 --validate True
python main.py --data vFashion --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 8 --batch_size 128 --validate True

# filters
python main.py --data vFashion --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 16 --batch_size 64 --validate True
python main.py --data vFashion --model gmsCNN --gpu 1 --kernels 7 --kernelg 3 --kernel 5 --num_filters 16 --batch_size 128 --validate True