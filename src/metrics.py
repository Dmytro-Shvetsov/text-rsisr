from numpy import zeros as np_zeros


def edit_distance(token1:str, token2:str) -> int:
    """
    Calculates edit distance aka Levenshtein distance between two strings.
    Returns:
        (int): calculated distance. 
    """

    n_rows = len(token1) + 1
    n_cols = len(token2) + 1
    distance_mat = np_zeros((n_rows, n_cols))
    for i in range(n_rows):
        distance_mat[i][0] = i
    
    for j in range(n_cols):
        distance_mat[0][j] = j
    
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            if token1[i - 1] == token2[j - 1]:
                distance_mat[i][j] = distance_mat[i - 1][j - 1]
            else:
                # unknown value of matrix is calculated from a window 2x2
                # say we have two tokens of length 2, so the iteration i=0, j=0 will deal with distance matrix 
                # 0 1 2
                # 1 x x
                # 2 x x
                # and mentioned 2x2 matrix will be
                # 0 1
                # 1 x
                l_bot = distance_mat[i][j - 1]
                r_top = distance_mat[i - 1][j]
                l_top = distance_mat[i - 1][j - 1]
                
                if l_bot <= r_top and l_bot <= l_top:
                    distance_mat[i][j] = l_bot + 1
                elif r_top <= l_bot and r_top <= l_top:
                    distance_mat[i][j] = r_top + 1
                else:
                    distance_mat[i][j] = l_top + 1
            
    return distance_mat[n_rows - 1][n_cols - 1]


def relative_distance(true_text, pred_text):
    return int(edit_distance(true_text, pred_text)) / len(true_text)
