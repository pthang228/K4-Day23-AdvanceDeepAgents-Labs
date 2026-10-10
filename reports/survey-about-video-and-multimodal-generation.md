# Survey on Video and Multimodal Generation

## TL;DR
- Hybrid VAE-GAN and diffusion models enable text- and image-conditioned video generation, producing realistic and diverse outputs [1].
- Recent advances in training-free adaptation allow high frame rate video generation without extensive training, improving efficiency [2].
- Despite progress, pretrained video models struggle with human intent alignment, temporal coherence, and physical plausibility, indicating a gap in real-world applicability [3].

## Background
Generative AI has become a transformative force in creating multimodal content, spanning text, vision, audio, video, and 3D/XR environments. However, existing surveys are fragmented, often focusing on a single modality or offering only qualitative assessments without systematic taxonomies or empirical validations. This lack of unified frameworks and objective benchmarks hinders comparative analysis and practical deployment across domains [4].

## Text- and Image-Conditioned Video Generation
Models such as VideoCrafter1 utilize diffusion architectures—specifically T2V and I2V—to generate high-quality videos from text and image inputs. T2V produces realistic cinematic videos, while I2V preserves the content of reference images, demonstrating strong fidelity in visual consistency. These models build upon hybrid VAE-GAN frameworks that generate text-conditioned videos, outperforming adapted text-to-image models in diversity and coherence [1][5].

## Efficiency and Real-Time Adaptation
ZeroSmooth introduces a training-free method for enhancing frame rates in generative diffusion models by integrating hidden state correction modules within a self-cascaded architecture. This approach achieves performance comparable to models trained with substantial computational resources, significantly reducing training overhead and enabling real-time video generation without retraining [2].

## Challenges in Human Intent and Temporal Coherence
While video generation has evolved from short, low-quality clips to long-duration, high-resolution sequences with complex spatiotemporal dynamics, pretrained models still face critical limitations. They often fail to reliably follow human intent, maintain temporal consistency, or adhere to physical laws and safety constraints. This gap underscores the need for better alignment mechanisms and robustness in real-world applications [3].

## Trends and open problems
Key trends include the rise of diffusion-based models for high-fidelity video generation and the development of training-free adaptation techniques to improve efficiency. Open problems remain in achieving reliable human intent alignment, ensuring temporal and physical coherence, and integrating safety constraints into generated content. Additionally, the absence of a unified taxonomy and empirical benchmarks in current literature limits the ability to compare and validate model performance across diverse scenarios [4].

## References
[1] Video Generation From Text. hf-search. https://huggingface.co/papers/1710.00421 (2017-10-01)
[2] ZeroSmooth: Training-free Diffuser Adaptation for High Frame Rate Video Generation. hf-search. https://huggingface.co/papers/2406.00908 (2024-06-03)
[3] Video Generation Models: A Survey of Post-Training and Alignment. arxiv. https://arxiv.org/abs/2610.00812 (2026-09-30)
[4] Generative AI for multimodal content: a survey with empirical and experimental evaluations | Artificial Intelligence Review | Springer Nature Link. web. https://link.springer.com/article/10.1007/s10462-026-11525-6 (2026-03-19)
[5] VideoCrafter1: Open Diffusion Models for High-Quality Video Generation. hf-search. https://huggingface.co/papers/2310.19512 (2023-10-30)
