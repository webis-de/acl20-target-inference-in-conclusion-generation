import nltk
import glob

nltk.download('punkt')

files_path  = './brat-project-essays/*.txt'
output_path = './corpus_sents.txt'

all_sents = []
for p in glob.glob(files_path):
	content = open(p).read()
	sents = nltk.sent_tokenize(content.decode('utf8'))
	all_sents = all_sents + sents

all_sents = list(map(lambda x: x.encode('utf8') + '\n', all_sents))
open(output_path, 'w').writelines(all_sents)