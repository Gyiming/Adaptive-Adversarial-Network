import os

def main():
    for i in range(99):
        cmd = 'mkdir ' + str(i)
        os.system(cmd)


if __name__ == '__main__':
    main()