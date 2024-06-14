import re
import math

import numpy as np
import pandas as pd
import os, sys

def readFasta(file):
	if os.path.exists(file) == False:
		print('Error: "' + file + '" does not exist.')
		sys.exit(1)

	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input file seems not in fasta format.')
		sys.exit(1)

	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta


def DDE(fastas, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'

	myCodons = {
		'A': 4,
		'C': 2,
		'D': 2,
		'E': 2,
		'F': 2,
		'G': 4,
		'H': 2,
		'I': 3,
		'K': 2,
		'L': 6,
		'M': 1,
		'N': 2,
		'P': 4,
		'Q': 2,
		'R': 6,
		'S': 6,
		'T': 4,
		'V': 4,
		'W': 1,
		'Y': 2

	}

	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#'] + diPeptides
	encodings.append(header)

	myTM = []
	for pair in diPeptides:
		myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]

		myTV = []
		for j in range(len(myTM)):
			myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

		for j in range(len(tmpCode)):
			tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

		code = code + tmpCode
		encodings.append(code)
	return encodings

kw=  {'path': r"D:\Coding\P1\AL\AL-Base\train.fasta",'order': 'ACDEFGHIKLMNPQRSTVWY'}
fastas1 = readFasta.readFasta(r"D:\Coding\P1\AL\AL-Base\train.fasta")

result=DDE(fastas1, **kw)

data1=np.matrix(result[1:])[:,1:]
data_=pd.DataFrame(data=data1)
data_.to_csv(r'D:\Coding\P1\AL\AL-Base\train_DDE.csv')
