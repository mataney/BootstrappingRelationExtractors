import argparse
import json
import requests

def main(args):
    headers = {'Content-Type': 'application/json'}
    
    with open(args.file_to_annotate, 'r') as f:
        texts = f.readlines()

    for i, text in enumerate(texts):
        payload = {'text': text}
        response = requests.post("http://localhost:9090/annotate-text", json=payload, headers=headers); json.loads(response.content)
        content = json.loads(response.content)
        sentences = content['sentences']
        out = ''
        for sent in sentences:
            out += ' '.join(sent['words'])
        out = out.replace('-LSB- ', '[')
        out = out.replace(' -RSB-', ']')
        print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_to_annotate", type=str, required=True)
    args = parser.parse_args()
    main(args)