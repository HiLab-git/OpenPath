from open_clip import create_model_from_pretrained, get_tokenizer
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config


if __name__ == "__main__":

    BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    # target_modules = ["visual.encoder.layers.{0}.qkv".format(i) for i in range(12)]
    target_modules = ['qkv']
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules, # ['k_proj', 'v_proj']
        lora_dropout=0.1,
        bias="none",
        # modules_to_save=["fc"],
        # modules_to_save=["fc"],
    )

    lora_model = get_peft_model(BMC_model, config)

    # print(list(BMC_model.named_children()))
    print(list(BMC_model.named_children()))
    BMC_model.visual