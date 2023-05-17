import json
import argparse


def main(args):

    retrieval_data = json.load(open(args.predictevid))
    classify_data = json.load(open(args.predictlabel))
    test_data = json.load(open("../train-claims.json"))

    unite_output = {}


    for claim_tag, claim_text in test_data.items():
        unite_output[claim_tag] = {"claim_label": classify_data[claim_tag], "claim_text": claim_text["claim_text"] , "evidences": retrieval_data[claim_tag]}
    
    
    default_outputdir = "./train-claims-predictions.json"
    output_dir = args.outputdir if args.outputdir is not None else default_outputdir
    
    f_out = open(output_dir, 'w')
    json.dump(unite_output, f_out)
    f_out.close()
    print("Done!")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Merging the retriever output and classifier output.")
    
    parser.add_argument("--predictevid", required=True)
    parser.add_argument("--predictlabel", required=True)
    parser.add_argument("--outputdir", required=False)
    args = parser.parse_args()
    
    main(args)