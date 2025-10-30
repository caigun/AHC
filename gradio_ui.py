import gradio as gr
from infer_and_compare import run_inference


def app_fn(context: str, question: str):
	res = run_inference(context or '', question or '')
	return (
		res['answer'],
		res['answer_html_teacher'],
		res['answer_html_embedded'],
	)


def build_app():
	with gr.Blocks(title="Embedded Hallucination Detector") as demo:
		gr.Markdown("**Embedded Hallucination Detector (generate with Qwen, then detect)**")
		with gr.Row():
			context = gr.Textbox(label="Context", lines=8, value="France is a country in Europe. The capital of France is Paris. The population of France is 67 million.")
			question = gr.Textbox(label="Question", lines=4, value="What is the capital of France? What is the population of France?")
		btn = gr.Button("Generate + Detect")
		gen_answer = gr.Textbox(label="Generated Answer", lines=6)
		with gr.Row():
			out_teacher = gr.HTML(label="LettuceDetect (teacher)")
			out_embedded = gr.HTML(label="Embedded (small encoder)")
		btn.click(app_fn, inputs=[context, question], outputs=[gen_answer, out_teacher, out_embedded])
	return demo


if __name__ == '__main__':
	demo = build_app()
	demo.launch()
