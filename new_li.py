from cryptography.hazmat.primitives import padding
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pickle
import random as r
import time

IV = os.urandom(16)
KEY = os.urandom(32)

def interval_to_prefix(a, b, pref, res):
    # pg 6
    #1
    k = 0
    while (k < len(a) and a[k] == b[k]):
        k += 1

    #2
    if (k == len(a)):
        res.append(pref + a)
        return res
    #3
    if (a[k:] == ''.join(['0' for i in range(0, len(a)-k)]) and b[k:] == ''.join(['1' for i in range(0, len(b)-k)])):
        # res.add(a[:k])
        res.append(pref + a[:k])
        return res

    #4 fix degeaba

    #5
    pref1 = pref + a[:k] + '0'
    pref2 = pref + a[:k] + '1'

    interval_to_prefix(a[k+1:], ''.join(['1' for i in range(0, len(a)-k-1)]), pref1, res)
    interval_to_prefix(''.join(['0' for i in range(0, len(a)-k-1)]), b[k+1:], pref2, res)

    return res


def start_interval(a, b):
    a = bin(a)[2:]
    b = bin(b)[2:]

    while (len(a)<16):
        a = '0' + a

    while (len(b)<16):
        b = '0' + b

    return interval_to_prefix(a,b, '', [])


def xor(a,b):
    if (a == b):
        return '0'
    return '1'


def encrypt(a, iv=IV, k=KEY):
    # a - string de 0 si 1
    a1 = a[0]
    for i in range(1, len(a)):
        # a1 = a1 + l(r(p(a[:i]), k))
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(bytes(a[:i], "utf-8")) + padder.finalize()
        backend = default_backend()
        cipher = Cipher(algorithms.AES(k), modes.CBC(iv), backend=backend)
        encryptor = cipher.encryptor()
        ct = encryptor.update(padded_data) + encryptor.finalize()
        integ = int.from_bytes(ct, byteorder='big')
        l = str(integ&1)
        a1 = a1 + xor(a[i], l)

    return a1


def encrypt_sub(sub):
    # transf sub in prefixe ->
    # -> sub de forma [(i, [pref1, pref2, etc.]), (i, [pref3, pref4, etc.]), ....]
    enc_sub = []
    for el in sub:
        enc_prefs = []
        prefs = start_interval(int(el[1][0]), int(el[1][1]))
        for p in prefs:
            enc_prefs.append(encrypt(p))

        enc_sub.append((el[0], enc_prefs))

    return enc_sub


def encrypt_pub(pub):
    enc_pub = []
    for el in pub:
        bin_el = bin(int(el))[2:]
        while(len(bin_el) < 16):
            bin_el = '0' + bin_el

        enc_pub.append(encrypt(bin_el))

    return enc_pub


def load_subs(filename):
    sub_file = filename + '_subs'
    f = open(sub_file, 'rb')

    enc_sub_file = filename + '_subs_enc'
    ef = open(enc_sub_file, 'wb')
    enc_subs = []

    while(True):
        try:
            sub = pickle.load(f)
            enc_sub = encrypt_sub(sub)
            # le scriu intr-un fisier?? echivalent le trimiti la broker in bd
            pickle.dump(enc_sub, ef)
            # le adaug intr-o lista, pentru test mai rapid
            enc_subs.append(enc_sub)

        except Exception as e:
            print(e)
            break

    f.close()
    ef.close()

    return enc_subs


def load_pubs(filename):
    pub_file = filename + '_pubs'
    f = open(pub_file, 'rb')

    enc_pub_file = filename + '_pubs_enc'
    ef = open(enc_pub_file, 'wb')
    enc_pubs = []

    while(True):
        try:
            pub = pickle.load(f)
            enc_pub = encrypt_pub(pub)
            # le scriu intr-un fisier?? echivalent le trimiti la broker in bd
            pickle.dump(enc_pub, ef)
            # le adaug intr-o lista, pentru test mai rapid
            enc_pubs.append(enc_pub)

        except Exception as e:
            print(e)
            break

    f.close()
    ef.close()

    return enc_pubs


def match_lists(subs, pubs):
    matches = []
    for pub in pubs:
        for sub in subs:
            is_match = []
            for s in sub:
                is_match.append('False')
                for pref in s[1]:
                    if (pub[s[0]][:len(pref)] == pref):
                        # la poz i e match
                        is_match[-1] = 'True'
            if (not 'False' in is_match):
                matches.append((pub, sub, 'true'))
            else:
                matches.append((pub, sub, 'false'))

    return matches


load_start = time.time()

enc_subs = load_subs('10000_6')
enc_pubs = load_pubs('10')

load_end = time.time()

load_time = load_end - load_start
print('Load time: ', load_time)

match_start = time.time()

for i in range(0, 10):
    matches = match_lists(enc_subs, enc_pubs)

match_end = time.time()

match_time = (match_end - match_start)/10
print('Match time: ', match_time)
