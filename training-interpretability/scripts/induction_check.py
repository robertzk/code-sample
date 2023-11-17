import os
import random
import torch
from training_interpretability.model import BasicTransformer, Config

def induction_check(model: torch.nn.Module) -> bool:
    random_seq = torch.tensor([random.sample(range(50257), k=20)*2]).type(torch.long) #.cuda()
    outputs = model(random_seq).argmax(dim=-1)
    pairs = list(zip(random_seq[:, 1:][0].tolist() + [0], outputs[0].tolist()))
    if (matches := sum(int(correct == predicted) for correct, predicted in pairs[random_seq.shape[1] // 2:])) >= random_seq.shape[1] // 2 * 0.5:
        return True
    else:
        return False

if __name__ == "__main__":
    model_config = Config(
        d_model=(d_model := 768 // 2),
        debug=False,
        layer_norm_epsilon=1e-5,
        d_vocab=50257,
        init_range=0.02,
        n_context=512,
        d_head=64,
        d_mlp=d_model * 2,
        n_heads=d_model // 64,
        n_layers=3,
        device="cpu"
    )

    print("Loading BasicTransformer")
    model = BasicTransformer(model_config)

    for file in os.listdir("experiments"):
        file = os.path.join("experiments", file)
        if not os.path.isfile(file):
            continue
        if file.endswith(".pt"):
            model_state = torch.load(file, map_location=torch.device('cpu'))
            model.load_state_dict(model_state)
            has_induction_heads = induction_check(model)
            print(f"{file} has induction heads: {has_induction_heads}")
