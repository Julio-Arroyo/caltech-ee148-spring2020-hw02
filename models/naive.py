def find_red(I):
    '''
    4 random images used to find tones of red: {213, 291, 251, }
    '''
    # red_tones = {(253,217,113), (240,67,97), (255,242,96), (253,168,113), }

    bboxes = []

    x_offset = 8
    y_offset = 6

    i = 0
    j = 0
    min_red = 235
    min_blue = 95
    max_blue = 115
    min_green = 65
    max_green = 245
    
    # use while loops instead of for loops to jump around image, sorry i know it's ugly
    while j < I.shape[1]:
        i = 0
        while i < 3 * (I.shape[0] // 5):  # look at top three-fifths of image only
            if (I[i, j, 0] > min_red
                and I[i, j, 1] > min_green and I[i, j, 1] < max_green
                and I[i, j, 2] > min_blue and I[i, j, 2] < max_blue):
                bboxes.append([j-x_offset+2, i-y_offset, j+x_offset+2, i+y_offset, 1])
                i = I.shape[0] - 2
                j += 20
            i += 1
        j += 1
    
    return bboxes
