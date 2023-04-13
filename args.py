#cach thay doi gia tri batch_size or num_epoch bang terminal 
import argparse

def get_args():
    parser = argparse.ArgumentParser(description= "Train a CNN Model")
    parser.add_argument("--batch_size", "-b1", type = int,default= 32 )
    parser.add_argument("--num_epochs", "-p", type = int , default= 100)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args.batch_size)
    print(args.num_epochs)