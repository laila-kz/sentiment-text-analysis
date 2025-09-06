from transformers import pipeline
#a pipline is a tool that bundles together a model + preprocessing + postprocessing.
#pipeline = takes text → cleans it → sends to model → returns result.
print("it is running... ")
#define a fct that will take a string and return the sentiment
def analyse(text: str):
    #create a sentiment analysis pipeline
    # a pipline loads a small pre-trained DistilBERT model fine-tuned on a dataset of movie reviews (positive/negative).
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier(text)[0]  #the classifier returns a list of dicts, we take the first one
    label = result['label'].capitalize()  #the label is either 'POSITIVE' or 'NEGATIVE'
    score= float(result["score"])  #the score is a float between 0 and 1
    #add a creative touch
    emoji = {"Positive":"😊","Negative":"😠"}.get(label, "😶")
    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))  #a simple bar to visualize the score
    return label, score, emoji, bar

if __name__ == "__main__":
    try:
        text = input("Enter text to analyze sentiment (or 'exit' to quit): ")
        if not text : #if the input is empty, we exit
            print("Please enter some text")
        else:
            label, score, emoji, bar = analyse(text)
            print(f"Result: {label} {emoji}")
            print(f"Confidence: {score:.2f}  {bar}")
    except KeyboardInterrupt:
        print("\nBye...!")


