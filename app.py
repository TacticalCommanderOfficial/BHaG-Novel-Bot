import os
import re
import gc
import datetime
import shutil
import torch
import gradio as gr
from unsloth import FastLanguageModel

# Base system directory path matching your Chromebook / Google Drive setup
DRIVE_BASE = "/content/drive/MyDrive/Archipelago_Edits"

def initialize_workspace(file_name):
    """Ensures base workspace paths and default manuscript baselines are securely mounted."""
    if not os.path.exists(DRIVE_BASE):
        os.makedirs(DRIVE_BASE, exist_ok=True)
    
    target_path = os.path.join(DRIVE_BASE, file_name)
    if not os.path.exists(target_path):
        # Generate an absolute baseline file containing multi-colored blocks for target testing
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("# Chapter 1: The Outpost Gate\n\n"
                    "> ### 🟢 GREEN: PRISTINE LINE\n"
                    "> Sliver adjusted his scales against the morning chill. His four claws scampered across the smooth basalt stone.\n\n"
                    "> ### 🔵 CYAN: DESCRIPTION BLOCK\n"
                    "> The Central Island Coalition trading hub was unusually active for a dawn shift.\n\n"
                    "> ### 🔴 RED: STRUCTURAL REWRITE\n"
                    "> Slipnotch wrapped his long coil tighter around the primary perimeter post, his tail giving an angry thrash.\n\n"
                    "> ### 🔵 CYAN: DESCRIPTION BLOCK\n"
                    "> Mist rolled heavy off the marshlands, carrying the scent of rotting reeds from the East outposts.")
    return target_path

def create_rolling_backup(target_path):
    """Safely duplicates the active document to a rolling backup directory before any edits."""
    try:
        if os.path.exists(target_path):
            backup_dir = os.path.join(DRIVE_BASE, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(target_path)
            shutil.copy2(target_path, os.path.join(backup_dir, f"bak_{timestamp}_{base_name}"))
            
            all_backups = sorted([os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.startswith("bak_")], key=os.path.getmtime)
            while len(all_backups) > 10: os.remove(all_backups.pop(0))
    except Exception: pass

# ==============================================================================
# INTELLECTUAL EXTRACTOR & INJECTION PARSING ENGINE
# ==============================================================================
def parse_and_modify_block(file_path, counting_instruction, slider_ctx, system_prompt, model, tokenizer, margin_notes, prg):
    """Scans the text file, isolates the requested color block index, executes AI edits, and replaces it natively."""
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    block_pattern = r"(> ### (🟢 GREEN|🔴 RED|🟠 ORANGE|🔵 CYAN):.*?\n(?:>.*?\n|\n)*)"
    blocks = [m.group(0) for m in re.finditer(block_pattern, full_text)]
    
    if not blocks:
        return "❌ Error: Could not find any formatted color blocks in the file.", full_text

    text_to_num = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7}
    norm_instruction = counting_instruction.lower()
    target_color = None
    for color in ["green", "red", "orange", "cyan"]:
        if color in norm_instruction: target_color = color.upper()
        
    target_rank = 1
    for word, num in text_to_num.items():
        if word in norm_instruction: target_rank = num

    matched_blocks = []
    for b in blocks:
        if f"### 🔴 RED" in b and target_color == "RED": matched_blocks.append(b)
        elif f"### 🔵 CYAN" in b and target_color == "CYAN": matched_blocks.append(b)
        elif f"### 🟠 ORANGE" in b and target_color == "ORANGE": matched_blocks.append(b)
        elif f"### 🟢 GREEN" in b and target_color == "GREEN": matched_blocks.append(b)

    if not matched_blocks or len(matched_blocks) < target_rank:
        return f"❌ Target Locator Missed: Could not locate a matching '{counting_instruction}'.", full_text

    target_raw_block = matched_blocks[target_rank - 1]
    lines = target_raw_block.split("\n")
    pure_story_lines = [re.sub(r"^>\s*", "", l) for l in lines if not l.startswith("> ###") and l.strip()]
    extracted_narrative = "\n".join(pure_story_lines)

    prg(0.6, desc="🧠 Framing AI workspace layers & running token model inference turns...")
    margin_context = f"[Editorial Instructions: {margin_notes}]" if margin_notes else ""
    
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{slider_ctx}\n{margin_context}\n\nRewrite this raw paragraph block:\n{extracted_narrative}"}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True, temperature=0.7, top_p=0.9)
    ai_raw_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape:], skip_special_tokens=True).strip()

    new_formatted_block = f"> ### 🔵 {target_color}: AUTOMATED REWRITE\n" if target_color == "CYAN" else f"> ### 🔴 {target_color}: AUTOMATED REWRITE\n"
    if target_color == "GREEN": new_formatted_block = f"> ### 🟢 GREEN: AUTOMATED REWRITE\n"
    if target_color == "ORANGE": new_formatted_block = f"> ### 🟠 ORANGE: AUTOMATED REWRITE\n"
    
    for line in ai_raw_text.split("\n"):
        new_formatted_block += f"> {line}\n"
    new_formatted_block += "\n"

    updated_file_contents = full_text.replace(target_raw_block, new_formatted_block)
    create_rolling_backup(file_path)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_file_contents)
        
    return f"✍️ Target chunk '{counting_instruction}' has been completely refactored.", updated_file_contents
    # ==============================================================================
# MODEL MANAGEMENT & GRADIO INTERFACE DISPATCH
# ==============================================================================
print("🤖 Booting optimized 4-bit Llama 3 8B Instruct model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit", max_seq_length=4096, dtype=None, load_in_4bit=True, device_map="cuda"
)
FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = """You are Calli Slidell, a world-class fiction editor and creative partner helping rewrite an epic fantasy novel manuscript titled "Archipelago: Shattered Scales".

LOCKED REPTILIAN BIOLOGY & CONTINUITY CONSTRAINTS:
1. THE ISLANDEES (Sliver, Speck, Twitch, Plume, Clearsnap): Quadruped LIZARDS/SKINKS. Four active claws, move by scampering/skittering, use snouts to nudge objects, adjust scales. Completely VENOMLESS. High metabolism protects against morning cold-slows.
2. THE MAINLAND MERCS (Slipnotch, Ash, Smolder): Literal SNAKES/VIPERS. Completely LEGLESS. Move strictly by slithering, winding, or thrashing. Coil up tightly into strike positions. Carry deadly VENOM and prominent fangs.
3. ECONOMY & TAX LOOP: Coalition work pays 20 shells. Independent lizards (Sliver, Speck) face a +30 shell penalty, paying 200 shells per five days tax. Private commerce inside Main Camp walls is strictly forbidden.
4. METANARRATIVE: Compiled by Clearsnap, who occasionally leaves meta-commentary annotations.

Transform the provided text snippet cleanly. Maintain active pacing (3.5/5). Return ONLY the rewritten narrative text."""

def process_automated_editorial_turn(file_name, block_instruction, atmosphere, mental, dialogue, margin_notes, prg=gr.Progress()):
    prg(0.1, desc="🔍 Locating manuscript file target index...")
    target_path = initialize_workspace(file_name)
    
    prg(0.3, desc="⚡ Reading file buffers and running structural parser...")
    slider_context = f"[Mode=Automated Turn, Atmosphere Sensory Detail={atmosphere}/5, Mental Depth={mental}/5, Dialogue Padding={dialogue}/5, Global Pacing=3.5/5]"
    
    log_output, updated_text = parse_and_modify_block(
        target_path, block_instruction, slider_context, SYSTEM_PROMPT, model, tokenizer, margin_notes, prg
    )
    
    prg(1.0, desc="🎨 Rerendering workspace dashboard views...")
    return log_output, updated_text

def read_document_contents(file_name):
    target_path = os.path.join(DRIVE_BASE, file_name)
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f: return f.read()
    return "Workspace session file empty."

def force_flush_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return "🧹 Memory Refresh Status: Complete. VRAM cells successfully flushed!"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦎 BHaG Novel Bot Development Studio — Calli Slidell Profile")
    gr.Markdown("*Autonomous file-indexing architectural build designed for hands-free ChromeOS operation.*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Automation Control Desk")
            target_file_input = gr.Textbox(value="manuscript.md", label="Active Workspace Target File Name")
            block_instruction_box = gr.Textbox(value="the second cyan block", label="🎯 Target Block Selector Instructions")
            
            slider_atmos = gr.Slider(1, 5, step=1, value=3, label="Atmosphere (Sensory Details)")
            slider_mental = gr.Slider(1, 5, step=1, value=3, label="Mental (Emotional & Logic Weight)")
            slider_dialogue = gr.Slider(1, 5, step=1, value=3, label="Dialogue (Padded Chatter)")
            
            margin_notes_box = gr.Textbox(lines=3, label="📝 Calli's Margin Notes", placeholder="Type instructions here...")
            submit_btn = gr.Button("🚀 Run Autonomous Document Edit", variant="primary")
            flush_btn = gr.Button("🧹 Refresh Memory Cells (Clear VRAM Cache)", variant="stop")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Execution & Activity Return Logs")
            log_monitor_box = gr.Textbox(lines=4, label="System Operations Feedback Pane", interactive=False)
            gr.Markdown("### 📜 Live Manuscript Viewport (`manuscript.md`)")
            file_viewport = gr.Markdown(value=read_document_contents("manuscript.md"))
            refresh_btn = gr.Button("🔄 Force Reload Viewport File State")

    submit_btn.click(
        fn=process_automated_editorial_turn, 
        inputs=[target_file_input, block_instruction_box, slider_atmos, slider_mental, slider_dialogue, margin_notes_box], 
        outputs=[log_monitor_box, file_viewport]
    )
    refresh_btn.click(fn=read_document_contents, inputs=[target_file_input], outputs=[file_viewport])
    flush_btn.click(fn=force_flush_vram, inputs=[], outputs=[log_monitor_box])

demo.launch(share=True, debug=True)

