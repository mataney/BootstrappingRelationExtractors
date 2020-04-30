import argparse
import json
import requests
from tqdm import tqdm

def main(args):
    headers = {'Content-Type': 'application/json'}
    
    with open(args.file_to_annotate, 'r') as f:
        texts = f.readlines()

    with open(args.file_to_annotate.split('.txt')[0]+'_good_tokenization.txt', 'w') as outfile:
        for i, text in tqdm(enumerate(texts)):
            payload = {'text': text}
            response = requests.post("http://localhost:9090/annotate-text", json=payload, headers=headers)
            content = json.loads(response.content)
            sentences = content['sentences']
            out = ''
            for sent in sentences:
                out += ' '.join(sent['words'])
            out = out.replace('-LSB- ', '[')
            out = out.replace(' -RSB-', ']')
            outfile.write(out+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_to_annotate", type=str, required=True)
    args = parser.parse_args()
    main(args)