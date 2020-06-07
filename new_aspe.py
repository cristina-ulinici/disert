import numpy as np
import pickle
import time

d = 10

def gen_M():
    M = np.random.randint(100, size=(d+1, d+1)) #genereaza numere intre 0 si 1
    M_1 = np.linalg.inv(M)

    MM_1 = np.dot(M,M_1)
    I = np.identity(d+1)

    while (not np.allclose(MM_1, I)):
        M = np.random.randint(100, size=(d+1, d+1)) #genereaza numere intre 0 si 1
        M_1 = np.linalg.inv(M)

        MM_1 = np.dot(M,M_1)
        I = np.identity(d+1)

    return M.transpose(), M_1


def encrypt_P(P):
    q = np.random.randint(100)
    P.append(1)
    P = np.array(P) * q
    PM_1 = np.matmul(M_1, P)

    return PM_1


def gen_S(i, v):
    S_i = []
    r = np.random.randint(100)
    for k in range(0, d):
        if (k == i):
            S_i.append(v)
        else:
            S_i.append(np.random.randint(100))

    S_i1 = S_i[:]
    S_i1[i] -= r

    S_i2 = S_i[:]
    S_i2[i] += r

    s1_new_el = -0.5*(np.linalg.norm(S_i1)*np.linalg.norm(S_i1))
    S_i1.append(s1_new_el)
    s2_new_el = -0.5*(np.linalg.norm(S_i2)*np.linalg.norm(S_i2))
    S_i2.append(s2_new_el)

    SMTi1 = np.matmul(M_T, np.array(S_i1))
    SMTi2 = np.matmul(M_T, np.array(S_i2))

    return (SMTi1, SMTi2)


def encrypt_S(S):
    ES = []
    for s in S:
        # s[1] = intervalul
        ES.append(s)
        if (s[1][0] == s[1][1]):
            v = gen_S(s[0], s[1][0])
            ES[-1][1][0] = v
            ES[-1][1][1] = v
        else:
            for a in s[1]:
                if (not (a == 0 or a == 65535)):
                    ES[-1][1][ES[-1][1].index(a)] = gen_S(s[0], a)
    return ES


def match_one_value(SMTi1, SMTi2, PM_1):
    v1 = np.subtract(SMTi2, SMTi1)
    rez = np.matmul(v1, PM_1)

    return rez


def match_lists(pubs, subs):
    matches = []
    for pub in pubs:
        for sub in subs:
            is_match = []
            for s in sub:
                # =
                if (len(s[1]) == 1):
                    rez = match_one_value(s[1][0][0],s[1][0][1], pub[s[0]])
                    is_match.append(np.isclose(rez, 0, atol=1e-05))

                # <
                elif (s[1][0] == 0):
                    rez = match_one_value(s[1][1][0],s[1][1][1], pub)
                    if (rez < 0):
                        is_match.append(np.bool_(True))
                    else:
                        is_match.append(np.bool_(False))

                # >
                elif (s[1][1] == 65535):
                    rez = match_one_value(s[1][0][0],s[1][0][1], pub)
                    if (rez > 0):
                        is_match.append(np.bool_(True))
                    else:
                        is_match.append(np.bool_(False))

                # [a, b]
                else:
                    rez1 = match_one_value(s[1][0][0],s[1][0][1], pub)
                    rez2 = match_one_value(s[1][1][0],s[1][1][1], pub)
                    if (rez1 > 0 and rez2 < 0):
                        is_match.append(np.bool_(True))
                    else:
                        is_match.append(np.bool_(False))

            if (np.bool_(False) not in is_match):
                matches.append((pub, sub, 'true'))
            else:
                matches.append((pub, sub, 'false'))
    return matches


def load_pubs(filename):
    pub_file = filename + '_pubs'
    f = open(pub_file, 'rb')

    enc_pub_file = filename + '_pubs_aspe_enc'
    ef = open(enc_pub_file, 'wb')
    enc_pubs = []

    while(True):
        try:
            pub = pickle.load(f)
            enc_pub = encrypt_P(pub)
            # le scriu intr-un fisier?? echivalent le trimiti la broker in bd
            pickle.dump(enc_pub, ef)
            # le adaug intr-o lista, pentru test mai rapid
            enc_pubs.append(enc_pub)


        except Exception as e:
            break

    f.close()
    ef.close()

    return enc_pubs


def load_subs(filename):
    sub_file = filename + '_subs'
    f = open(sub_file, 'rb')

    enc_sub_file = filename + '_subs_aspe_enc'
    ef = open(enc_sub_file, 'wb')
    enc_subs = []

    while(True):
        try:
            sub = pickle.load(f)
            enc_sub = encrypt_S(sub)
            # le scriu intr-un fisier?? echivalent le trimiti la broker in bd
            pickle.dump(enc_sub, ef)
            # le adaug intr-o lista, pentru test mai rapid
            enc_subs.append(enc_sub)


        except Exception as e:
            break

    f.close()
    ef.close()

    return enc_subs


M_T, M_1 = gen_M()

load_start = time.time()

enc_pubs = load_pubs('10')
enc_subs = load_subs('10000_10')

load_end = time.time()

load_time = load_end - load_start
print('Load time: ', load_time)


match_start = time.time()

for i in range(0,10):
    matches = match_lists(enc_pubs, enc_subs)

match_end = time.time()

match_time = (match_end - match_start)/10

print('Match_time: ', match_time)
