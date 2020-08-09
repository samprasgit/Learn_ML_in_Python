try:
    f = open('将进酒.txt', 'w')
    for line in f:
        print(line)
except OSError as error:
    print("出错啦！%s" % str(error))

finally:
    f.close()
