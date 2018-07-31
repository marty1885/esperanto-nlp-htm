import numpy as np
import sys

tokens = []

#List of esperanto alphabets
esperanto_cap_alphabet = ['A', 'B', 'C', 'Ĉ', 'D', 'E', 'F', 'G', 'Ĝ', 'H', 'Ĥ', 'I', 'J',
        'Ĵ', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'Ŝ', 'T', 'U', 'Ŭ', 'Z']
esperanto_low_alphabet = ['a', 'b', 'c', 'ĉ', 'd', 'e', 'f', 'g', 'ĝ', 'h', 'ĥ', 'i', 'j',
        'ĵ', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 'ŝ', 't', 'u', 'ŭ', 'v', 'z']

def esperantoToLower(ch):
    if ch in esperanto_cap_alphabet:
        return esperanto_low_alphabet[esperanto_cap_alphabet.index(ch)]
    return ch

def charToToken(ch):
    c = esperantoToLower(ch)
    base_len = len(esperanto_low_alphabet)+1
    additional_chars = [' ']
    if c == '.':
        return 0 #. is a special token
    elif c in esperanto_low_alphabet:
        return esperanto_low_alphabet.index(c) + 1
    elif c in additional_chars:
        return base_len+additional_chars.index(c)

tokens = []
with open(sys.argv[1]) as f:
    for line in f:
        curr_token = []
        for ch in line:
            token = charToToken(ch)
            if token != None:
                curr_token.append(token)
        tokens.extend(curr_token)

np.save(sys.argv[2], np.array(tokens, dtype=np.int32))