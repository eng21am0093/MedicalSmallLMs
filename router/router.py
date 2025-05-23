import pandas as pd
import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from openai import OpenAI
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

def parse_quality(evaluation_text: str) -> float:
    scores = [int(s) for s in re.findall(r'(\d+)', str(evaluation_text))]
    return sum(scores) / len(scores) if scores else 0.0



def prepare_data(gpt_path: str, meerkat_path: str, complexity_threshold: float = 0.0):
    # Read metric spreadsheets (header row at index 1)
    df_gpt = pd.read_excel(gpt_path, header=1)
    df_meerkat = pd.read_excel(meerkat_path, header=1)

    # Forward-fill the original prompt column
    df_gpt['prompt'] = df_gpt['Unnamed: 0'].ffill()
    df_meerkat['prompt'] = df_meerkat['Unnamed: 0'].ffill()

    # Parse numeric quality scores
    df_gpt['quality_gpt'] = df_gpt['evaluation'].apply(parse_quality)
    df_meerkat['quality_meerkat'] = df_meerkat['evaluation'].apply(parse_quality)

    # Merge on variant prompts
    df = pd.merge(
        df_gpt[['prompt', 'variant_prompt', 'quality_gpt']],
        df_meerkat[['variant_prompt', 'quality_meerkat']],
        on='variant_prompt'
    )

    # Label as "complex" when Meerkat falls behind GPT by threshold
    df['is_complex'] = df['quality_meerkat'] < df['quality_gpt'] - complexity_threshold

    # Embed prompts via OpenAI
    client = OpenAI()
    embed_resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=df['variant_prompt'].tolist()
    )
    embeddings = [choice.embedding for choice in embed_resp.data]

    return embeddings, df['is_complex']

def train_router(gpt_metrics_path: str, meerkat_metrics_path: str, threshold: float = 0.0):
    X, y = prepare_data(gpt_metrics_path, meerkat_metrics_path, threshold)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Validation Accuracy: {val_acc:.3f}")
    with open("router.pkl", "wb") as f:
        pickle.dump(clf, f)
    # Return all three values
    return clf, X_test, y_test


class RouterService:
    def __init__(
        self,
        router_path: str = "router.pkl",
        embedding_model: str = "text-embedding-3-small",
        gpt_model: str = "gpt-4o",
    ):
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        # load sklearn router
        with open(router_path, "rb") as f:
            self.router = pickle.load(f)
        # init local Meerkat client
        #self.meerkat = MeerkatClient(model_name="dmis-lab/meerkat-7b-v1.0")

    def route(self, prompt: str) -> str:
        emb_resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=prompt
        )
        embedding = emb_resp.data[0].embedding
        use_gpt = self.router.predict([embedding])[0]
        return "llama" if use_gpt else "meerkat"

if __name__ == "__main__":
    clf, X_test, y_test = train_router(
        gpt_metrics_path=r"",
        meerkat_metrics_path=r"",
        threshold=0.5
    )

    print(confusion_matrix(y_test, clf.predict(X_test)))
    service = RouterService()
    prompt = "Explain the pathophysiology of ALS in simple terms."
    choice = service.route(prompt)
    print(choice)
