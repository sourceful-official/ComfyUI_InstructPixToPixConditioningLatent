import torch

class InstructPixToPixConditioningLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "latent_image": ("LATENT", ),  # Input for pre-encoded latent
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/instructpix2pix"

    def encode(self, positive, negative, latent_image):
        # The input latent_image is now the encoded image
        concat_latent = latent_image

        out_latent = {}
        out_latent["samples"] = concat_latent  # Directly use the input latent

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], out_latent)

NODE_CLASS_MAPPINGS = {
    "InstructPixToPixConditioningLatent": InstructPixToPixConditioningLatent,
}