from dataclasses import dataclass
import os
import random
import torch
import tqdm

from easy_transformer import EasyTransformer
from transformers import AutoTokenizer
from training_interpretability.data import PileLoader
from training_interpretability.model import BasicTransformer, Config
from training_interpretability.gradient_recorder import BasicFileGradientRecorder
from training_interpretability.tokenizer import GPT2Tokenizer
from training_interpretability.train import Trainer, CheckpointModelConfig

@dataclass
class TrainConfig:
    grads_dir: str
    checkpoints_dir: str


def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randn(shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randint(100, 1000, shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output

def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    # Allow inputs of strings or tensors
    if isinstance(input_name, str): 
        reference_input = cache_dict[input_name]
    else:
        reference_input = input_name
    print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    print("Output shape:", output.shape)
    reference_output = gpt2_layer(reference_input)
    print("Reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct")
    return output

if __name__ == "__main__":
    model_config = Config(
        d_model=(d_model := 768 // 2),
        debug=False,
        layer_norm_epsilon=1e-5,
        d_vocab=50257,
        init_range=0.02,
        n_context=1024,
        d_head=64,
        d_mlp=d_model * 2,
        n_heads=d_model // 64,
        n_layers=3,
        device="cuda"
    )

    train_config = TrainConfig(
        grads_dir="experiments/induction-heads-003/grads",
        checkpoints_dir="experiments/induction-heads-003/checkpoints",
    )

    checkpoint_model_config = CheckpointModelConfig(
        checkpoint_dir=train_config.checkpoints_dir,
        checkpoint_interval=100,
    )

    print("Loading BasicTransformer")
    model = BasicTransformer(model_config).to("cuda")

    print("Loading reference tokenizer and supporting objects")
    tokenizer = GPT2Tokenizer(AutoTokenizer.from_pretrained("gpt2"))

    gradient_recorder = BasicFileGradientRecorder(
        log_single_datums=True,
        exclude_units=("embed.W_E", "pos_embed.W_pos", "unembed.W_U", "unembed.b_U"),
        apply_grad_sanity_check=False,
        dir_path=train_config.grads_dir,
    )

    batch_size = 25
    context_length = model_config.n_context
    data_loader = PileLoader(batch_size, context_length, tokenizer)
    target_tokens = 3_000_000_000
    train_steps = target_tokens // (batch_size * context_length)
    print(f"Training for {train_steps} train steps")

    trainer = Trainer(model, tokenizer, data_loader, train_steps=train_steps, d_vocab=model_config.d_vocab,
                      lr=10e-3, lr_step=1000, lr_gamma=0.03, clip_grad_norm=0.5,
                      gradient_recorder=gradient_recorder, checkpoint_model_config=checkpoint_model_config)
    trainer.train((batch_size, model_config.n_context), model_config.device)

    try:
        os.mkdir(train_config.checkpoints_dir, exist_ok=False)
    except OSError as e:
        pass
    model_params_path = os.path.join(train_config.checkpoints_dir, f"{train_steps}.pt")
    torch.save(model.state_dict(), model_params_path)

    try:
        random_seq = torch.Tensor([random.sample(range(model.cfg.d_vocab), k=20)*2]).type(torch.long).cuda()
        outputs = model(random_seq).argmax(dim=-1)
        pairs = list(zip(random_seq[:, 1:][0].tolist() + [0], outputs[0].tolist()))
        if (matches := sum(int(correct == predicted) for correct, predicted in pairs[random_seq.shape[1] // 2:])) >= random_seq.shape[1] // 2 * 0.75:
            print(f"Induction heads detected! Random sequence duplication predicted at {matches / (random_seq.shape[1] // 2) * 100:.0f}%")
        else:
            print("Induction heads not detected!")
    finally:
        import pdb; pdb.set_trace()

if __name__ == "__main__" and False:

    model_config = Config(
        d_model=768,
        debug=False,
        layer_norm_epsilon=1e-5,
        d_vocab=50257,
        init_range=0.02,
        n_context=1024,
        d_head=64,
        d_mlp=3072,
        n_heads=12,
        n_layers=12
    )

    print("Loading BasicTransformer")
    model = BasicTransformer(model_config)

    print("Loading reference model")
    reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = reference_gpt2.to_tokens(reference_text).cuda()
    logits, cache = reference_gpt2.run_with_cache(tokens)
    
    print("Importing reference model to BasicTransformer state")
    model.load_state_dict(reference_gpt2.state_dict(), strict=False)
    model.cuda()

    test_string = """Mini scule is a species of microhylid frog endemic to Madagascar that was described in 2019. The scientific name of the species refers to its size, being a pun on the word minuscule. It is very small, measuring only 8.4 to 10.8 mm (0.33 to 0.43 in) in snout–vent length. It has bronze underparts with a brown groin and back of the thigh, cream upperparts with brown flecking, a dark brown side of the head, and a red iris. On the hind feet, the first toe is absent and the second and fifth toes are strongly reduced. The frog is known only from the Sainte Luce Reserve, where it inhabits areas with deep leaf litter near semi-permanent water bodies. Specimens of frogs from Mandena, the Vohimena mountains, the southern Anosy Mountains, and Tsitongambarika may also be of this species. Along with Mini mum and Mini ature, the other two species in its genus, it received media attention when first described due to the wordplay in its scientific name. (Full article...)"""

    test_tokens = reference_gpt2.to_tokens(test_string).cuda()
    demo_logits = model(test_tokens)
    print(f"Test string output: {reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())}")
    
    print(f"Test tokens shape: {test_tokens.shape}")

    loss = lm_cross_entropy_loss(demo_logits, test_tokens)
    print(loss)
    print(f"Loss as average prob: {(-loss).exp()}")

    generate_text = False
    if generate_text:
        print("Generating some text:")

        test_string = "Breaking News: President Trump has been impeached by the House of Representatives for abuse of power and obstruction of Congress. The vote was 230 to 197, with 10 Republicans joining all Democrats in voting to impeach. The president is now only the third in American history to be impeached, and the first to be impeached twice. The House will now send the articles of impeachment to the Senate, where a trial will be held to determine whether to remove the president from office. The Senate is expected to begin the trial on"

        for i in tqdm.tqdm(range(100)):
            test_tokens = reference_gpt2.to_tokens(test_string).cuda()
            demo_logits = model(test_tokens)
            test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
    
    check_induction_heads = True
    if check_induction_heads:
        random_seq = torch.Tensor([random.sample(range(model.cfg.d_vocab), k=20)*2]).type(torch.int32).cuda()
        outputs = model(random_seq).argmax(dim=-1)
        pairs = list(zip(random_seq[:, 1:][0].tolist() + [0], outputs[0].tolist()))
        if (matches := sum(int(correct == predicted) for correct, predicted in pairs[random_seq.shape[1] // 2:])) >= random_seq.shape[1] // 2 * 0.75:
            print(f"Induction heads detected! Random sequence duplication predicted at {matches / (random_seq.shape[1] // 2) * 100:.0f}%")


        import pdb; pdb.set_trace()
        #zip(random_seq[1:], outputs)

    
    
    print(test_string)

    
def tests():
    from training_interpretability.model import LayerNorm, Embed, PositionalEmbedding, Attention, MLP, TransformerBlock, Unembed
    _ = rand_float_test(LayerNorm, [2, 4, 768])
    _ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post", cache.cache_dict)
    rand_int_test(Embed, [2, 4])
    load_gpt2_test(Embed, reference_gpt2.embed, tokens, cache.cache_dict)
    rand_int_test(PositionalEmbedding, [2, 4])
    load_gpt2_test(PositionalEmbedding, reference_gpt2.pos_embed, tokens, cache.cache_dict)
    rand_float_test(Attention, [2, 4, 768])
    load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"], cache.cache_dict)
    rand_float_test(MLP, [2, 4, 768])
    load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"], cache.cache_dict)
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0], cache.cache_dict)
    rand_float_test(Unembed, [2, 4, 768])
    load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"], cache.cache_dict)
    rand_int_test(BasicTransformer, [2, 4])
    load_gpt2_test(BasicTransformer, reference_gpt2, tokens, cache.cache_dict)
