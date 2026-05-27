# Third-Party Licenses

HeXA is distributed under the MIT License (see [LICENSE](LICENSE)). It bundles or depends on the following third-party software, each governed by its own license. This file lists the major dependencies for attribution and license compliance. Each project's full license text is available in its own repository.

## Bundled in this repository

| Project | Location | License | Source |
|---|---|---|---|
| InterPhyre | [`interphyre/`](interphyre/) | MIT | https://github.com/sankaranv/interphyre |

The InterPhyre simulator is included as a subdirectory under `interphyre/`. Its own LICENSE file is preserved at `interphyre/LICENSE`.

## Runtime Python dependencies

Installed via `pip install -r requirements.txt` and `pip install ./interphyre`.

| Package | License | Project URL |
|---|---|---|
| numpy | BSD-3-Clause | https://numpy.org |
| opencv-python | Apache-2.0 | https://github.com/opencv/opencv-python |
| Pillow | HPND (MIT-style) | https://python-pillow.org |
| matplotlib | PSF-based (matplotlib license) | https://matplotlib.org |
| tiktoken | MIT | https://github.com/openai/tiktoken |
| tqdm | MIT + MPL-2.0 | https://github.com/tqdm/tqdm |
| box2d-py | zlib | https://github.com/openai/box2d-py |
| gymnasium | MIT | https://github.com/Farama-Foundation/Gymnasium |
| pygame | LGPL-2.1 | https://www.pygame.org |

## Optional dependencies (OSS / Qwen baselines only)

These are not installed by default. They are needed only for the Qwen / GPT-OSS variants of `react_agent/run_react.py`:

| Package | License | Project URL |
|---|---|---|
| torch (PyTorch) | BSD-3-Clause | https://pytorch.org |
| transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| vllm | Apache-2.0 | https://github.com/vllm-project/vllm |

## External services

HeXA uses the [Claude Code CLI](https://claude.ai/code) as an external subprocess for the Claude actor and teacher models. The CLI itself is distributed by Anthropic; usage is subject to Anthropic's terms of service. HeXA does **not** ship or redistribute any Anthropic software — it invokes the `claude` binary that the user installs separately.

## Model attributions

When using the OSS variants, the following model weights are downloaded from Hugging Face under their respective licenses:

| Model | License |
|---|---|
| `Qwen/Qwen2.5-{7B,14B,32B}-Instruct` | Apache-2.0 (per Qwen2.5 license) |
| `openai/gpt-oss-{20b,120b}` | Apache-2.0 |

## Notice

Each third-party project's full license text is the authoritative source. The table above is provided for convenience only and does not modify or supersede those licenses.

If you redistribute HeXA, please retain this `THIRD_PARTY_LICENSES.md` and the `interphyre/LICENSE` file.
