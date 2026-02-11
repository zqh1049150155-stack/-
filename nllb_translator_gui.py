import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import threading
import psutil
import os
import sys

# 检查sentencepiece是否安装
try:
    import sentencepiece
except ImportError:
    print("错误：缺少sentencepiece库！")
    print("请运行：pip install sentencepiece")
    sys.exit(1)

class NLLBTranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NLLB-200 翻译工具")
        self.root.geometry("1000x700")
        
        # 使用Hugging Face模型ID，transformers会自动使用本地缓存
        self.model_id = "facebook/nllb-200-distilled-1.3B"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_overlap_chars = 80
        
        # 创建界面
        self.create_widgets()
        
        # 启动内存监控（使用独立线程）
        self.monitoring = True
        self.start_memory_monitor_thread()
        
        # 自动加载模型（延迟500ms，等界面显示完成）
        self.root.after(500, self.load_model_thread)
        
    def create_widgets(self):
        # 顶部状态栏
        status_frame = tk.Frame(self.root, bg="#f0f0f0", height=40)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 设备信息
        device_label = tk.Label(status_frame, text=f"设备: {self.device.upper()}", 
                               bg="#f0f0f0", font=("Arial", 10, "bold"))
        device_label.pack(side=tk.LEFT, padx=10)
        
        # 显存/内存状态
        self.memory_label = tk.Label(status_frame, text="内存: 加载中...", 
                                     bg="#f0f0f0", font=("Arial", 10))
        self.memory_label.pack(side=tk.RIGHT, padx=10)
        
        # 警告标签
        self.warning_label = tk.Label(status_frame, text="", 
                                     bg="#f0f0f0", fg="red", font=("Arial", 10, "bold"))
        self.warning_label.pack(side=tk.RIGHT, padx=10)
        
        # 语言选择区域
        lang_frame = tk.Frame(self.root)
        lang_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 统一的语言列表（添加"自动检测"选项）
        self.language_options = ('自动检测', '中文 (zho_Hans)', '英语 (eng_Latn)', '日语 (jpn_Jpan)', 
                                 '韩语 (kor_Hang)', '法语 (fra_Latn)', '德语 (deu_Latn)',
                                 '西班牙语 (spa_Latn)', '俄语 (rus_Cyrl)')
        
        tk.Label(lang_frame, text="源语言:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.source_lang = ttk.Combobox(lang_frame, width=15, state="readonly")
        self.source_lang['values'] = self.language_options
        self.source_lang.current(0)  # 默认自动检测
        self.source_lang.pack(side=tk.LEFT, padx=5)
        
        # 对调按钮
        swap_button = tk.Button(lang_frame, text="⇄", font=("Arial", 14, "bold"), 
                               command=self.swap_languages, width=3, height=1,
                               bg="#4CAF50", fg="white", cursor="hand2")
        swap_button.pack(side=tk.LEFT, padx=10)
        
        tk.Label(lang_frame, text="目标语言:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.target_lang = ttk.Combobox(lang_frame, width=15, state="readonly")
        # 目标语言不包含"自动检测"
        self.target_language_options = ('中文 (zho_Hans)', '英语 (eng_Latn)', '日语 (jpn_Jpan)', 
                                        '韩语 (kor_Hang)', '法语 (fra_Latn)', '德语 (deu_Latn)',
                                        '西班牙语 (spa_Latn)', '俄语 (rus_Cyrl)')
        self.target_lang['values'] = self.target_language_options
        self.target_lang.current(0)  # 默认中文
        self.target_lang.pack(side=tk.LEFT, padx=5)
        
        # 主内容区域
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧 - 输入区域
        left_frame = tk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(left_frame, text="输入文本:", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, 
                                                    font=("Arial", 11), height=20)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 右侧 - 输出区域
        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(right_frame, text="翻译结果:", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                                     font=("Arial", 11), height=20,
                                                     state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 底部按钮区域
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.load_button = tk.Button(button_frame, text="加载模型", 
                                     command=self.load_model_thread,
                                     font=("Arial", 11, "bold"), bg="#4CAF50", fg="white",
                                     width=12, height=2)
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        self.translate_button = tk.Button(button_frame, text="翻译", 
                                         command=self.translate_thread,
                                         font=("Arial", 11, "bold"), bg="#2196F3", fg="white",
                                         width=12, height=2, state=tk.DISABLED)
        self.translate_button.pack(side=tk.LEFT, padx=10)
        
        self.segment_translate_button = tk.Button(button_frame, text="分段翻译", 
                                                 command=self.segment_translate_thread,
                                                 font=("Arial", 11, "bold"), bg="#FF9800", fg="white",
                                                 width=12, height=2, state=tk.DISABLED)
        self.segment_translate_button.pack(side=tk.LEFT, padx=10)

        # 上下文连接选项（小幅增加显存占用，提升分段翻译连贯性）
        self.context_mode_var = tk.BooleanVar(value=True)
        self.context_mode_check = tk.Checkbutton(
            button_frame,
            text="上下文连接",
            variable=self.context_mode_var,
            font=("Arial", 10),
            onvalue=True,
            offvalue=False
        )
        self.context_mode_check.pack(side=tk.LEFT, padx=8)
        
        self.clear_button = tk.Button(button_frame, text="清空", 
                                     command=self.clear_text,
                                     font=("Arial", 11, "bold"), bg="#f44336", fg="white",
                                     width=12, height=2)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text=f"就绪 | 模型: {self.model_id}", 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def get_memory_info(self):
        """获取内存和显存信息"""
        try:
            # CPU内存
            ram = psutil.virtual_memory()
            ram_used = ram.used / (1024**3)  # GB
            ram_total = ram.total / (1024**3)  # GB
            ram_percent = ram.percent
            
            info = f"内存: {ram_used:.1f}/{ram_total:.1f}GB ({ram_percent:.1f}%)"
            warning = ""
            
            # GPU显存
            if torch.cuda.is_available():
                try:
                    # 使用mem_get_info获取实时显存信息（更准确）
                    mem_free, mem_total = torch.cuda.mem_get_info(0)
                    mem_used = mem_total - mem_free
                    mem_used_gb = mem_used / (1024**3)
                    mem_total_gb = mem_total / (1024**3)
                    mem_percent = (mem_used / mem_total) * 100
                    
                    info = f"显存: {mem_used_gb:.1f}/{mem_total_gb:.1f}GB ({mem_percent:.1f}%) | {info}"
                    
                    # 显存不足警告
                    if mem_percent > 90:
                        warning = "⚠️ 显存不足！"
                    elif mem_percent > 80:
                        warning = "⚠️ 显存紧张！"
                except Exception as e:
                    # 如果获取失败，显示错误
                    info = f"显存: 获取失败 | {info}"
            
            # 内存不足警告
            if ram_percent > 90:
                if warning:
                    warning = "⚠️ 显存和内存不足！"
                else:
                    warning = "⚠️ 内存不足！"
            elif ram_percent > 80 and not warning:
                warning = "⚠️ 内存紧张！"
                
            return info, warning
        except Exception as e:
            return f"内存监控错误: {str(e)}", ""
    
    def start_memory_monitor_thread(self):
        """启动内存监控线程"""
        def monitor_loop():
            import time
            while self.monitoring:
                try:
                    info, warning = self.get_memory_info()
                    # 使用after确保在主线程更新GUI
                    self.root.after(0, lambda: self.memory_label.config(text=info))
                    self.root.after(0, lambda: self.warning_label.config(text=warning))
                except Exception as e:
                    error_msg = f"监控错误: {str(e)[:50]}"
                    self.root.after(0, lambda: self.memory_label.config(text=error_msg))
                    print(f"监控错误: {e}")
                
                time.sleep(1)  # 每秒刷新
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def monitor_memory(self):
        """定时监控内存（旧版本，已废弃）"""
        try:
            info, warning = self.get_memory_info()
            self.memory_label.config(text=info)
            self.warning_label.config(text=warning)
            # 可选：在控制台输出，确认在刷新（调试用）
            # print(f"[刷新] {info}")
        except Exception as e:
            # 即使出错也显示错误信息，不中断刷新
            self.memory_label.config(text=f"监控错误: {str(e)[:50]}")
            print(f"监控错误: {e}")
        
        # 每秒更新一次 - 无论如何都继续
        self.root.after(1000, self.monitor_memory)
    
    def detect_language(self, text):
        """
        轻量级语言检测（基于Unicode范围和关键词）
        不需要额外的库，性能开销极小
        """
        # 统计不同字符集的数量
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 256)
        
        total_chars = len(text.strip())
        if total_chars == 0:
            return 'eng_Latn'  # 默认英语
        
        # 计算占比
        chinese_ratio = chinese_chars / total_chars
        japanese_ratio = japanese_chars / total_chars
        korean_ratio = korean_chars / total_chars
        cyrillic_ratio = cyrillic_chars / total_chars
        latin_ratio = latin_chars / total_chars
        
        # 根据占比判断（阈值0.3）
        if chinese_ratio > 0.3:
            return 'zho_Hans'
        elif japanese_ratio > 0.2:
            return 'jpn_Jpan'
        elif korean_ratio > 0.3:
            return 'kor_Hang'
        elif cyrillic_ratio > 0.3:
            return 'rus_Cyrl'
        elif latin_ratio > 0.5:
            # 拉丁字母可能是英语、法语、德语、西班牙语等
            # 简单判断：检查常见词
            text_lower = text.lower()
            if any(word in text_lower for word in ['the', 'is', 'and', 'of', 'to', 'a', 'in']):
                return 'eng_Latn'
            elif any(word in text_lower for word in ['le', 'la', 'les', 'de', 'et', 'un', 'une']):
                return 'fra_Latn'
            elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist', 'ein', 'eine']):
                return 'deu_Latn'
            elif any(word in text_lower for word in ['el', 'la', 'los', 'las', 'de', 'y', 'un', 'una']):
                return 'spa_Latn'
            else:
                return 'eng_Latn'  # 默认英语
        else:
            return 'eng_Latn'  # 默认英语
    
    def swap_languages(self):
        """对调源语言和目标语言"""
        # 获取当前选择的值
        source_value = self.source_lang.get()
        target_value = self.target_lang.get()
        
        # 如果源语言是"自动检测"，不进行对调
        if source_value == '自动检测':
            messagebox.showinfo("提示", "源语言为自动检测时无法对调")
            return
        
        # 交换值
        self.source_lang.set(target_value)
        self.target_lang.set(source_value)
        
        self.status_bar.config(text="已对调语言")
        print(f"对调: {source_value} ⇄ {target_value}")  # 调试输出
    
    def load_model_thread(self):
        """在新线程中加载模型"""
        thread = threading.Thread(target=self.load_model)
        thread.daemon = True
        thread.start()
    
    def load_model(self):
        """加载NLLB模型"""
        try:
            self.status_bar.config(text="正在加载模型...")
            self.load_button.config(state=tk.DISABLED, text="加载中...")
            
            # 检查sentencepiece
            try:
                import sentencepiece
            except ImportError:
                error_msg = "缺少必要的依赖库！\n\n请先安装 sentencepiece：\n\npip install sentencepiece\n\n或者:\n\npip install sentencepiece protobuf"
                messagebox.showerror("依赖缺失", error_msg)
                self.load_button.config(state=tk.NORMAL, text="加载模型")
                self.status_bar.config(text="缺少sentencepiece库")
                return
            
            # 加载tokenizer - 使用模型ID，transformers会自动使用缓存
            self.status_bar.config(text="加载Tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_fast=False,
                    legacy=False
                )
            except Exception as e1:
                error_msg = f"Tokenizer加载失败！\n\n错误: {str(e1)[:200]}"
                messagebox.showerror("Tokenizer错误", error_msg)
                self.load_button.config(state=tk.NORMAL, text="重新加载")
                self.status_bar.config(text="Tokenizer加载失败")
                raise
            
            # 加载模型，使用float16减少显存
            self.status_bar.config(text="加载模型 (使用FP16精度)...")
            
            # 检查PyTorch版本
            torch_version = torch.__version__.split('+')[0]
            major, minor = map(int, torch_version.split('.')[:2])
            
            if major < 2 or (major == 2 and minor < 6):
                warning_msg = f"检测到PyTorch版本 {torch_version}\n\n推荐升级到2.6+以获得更好的性能和安全性：\n\npip install torch --upgrade\n\n是否继续使用当前版本加载？"
                if not messagebox.askyesno("版本提示", warning_msg):
                    self.load_button.config(state=tk.NORMAL, text="加载模型")
                    self.status_bar.config(text="用户取消")
                    return
            
            if self.device == "cuda":
                # 使用float16精度，设置weights_only=True避免安全警告
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).to(self.device)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id,
                    low_cpu_mem_usage=True
                )
            
            self.status_bar.config(text="模型加载成功！")
            self.load_button.config(text="模型已加载", bg="#808080")
            self.translate_button.config(state=tk.NORMAL)
            self.segment_translate_button.config(state=tk.NORMAL)
            
            messagebox.showinfo("成功", "模型加载成功！\n已启用FP16精度以减少显存使用。")
            
        except torch.cuda.OutOfMemoryError:
            error_msg = "显存不足！无法加载模型。\n\n建议：\n1. 关闭其他占用显存的程序\n2. 使用更小的模型\n3. 增加虚拟内存"
            messagebox.showerror("显存不足", error_msg)
            self.status_bar.config(text="加载失败：显存不足")
            self.load_button.config(state=tk.NORMAL, text="重新加载")
            self.warning_label.config(text="⚠️ 显存不足！")
            
        except MemoryError:
            error_msg = "内存不足！无法加载模型。\n\n建议：\n1. 关闭其他占用内存的程序\n2. 增加虚拟内存"
            messagebox.showerror("内存不足", error_msg)
            self.status_bar.config(text="加载失败：内存不足")
            self.load_button.config(state=tk.NORMAL, text="重新加载")
            self.warning_label.config(text="⚠️ 内存不足！")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败:\n{str(e)}")
            self.status_bar.config(text="加载失败")
            self.load_button.config(state=tk.NORMAL, text="重新加载")
    
    def translate_thread(self):
        """在新线程中进行翻译"""
        thread = threading.Thread(target=self.translate)
        thread.daemon = True
        thread.start()
    
    def segment_translate_thread(self):
        """在新线程中进行分段翻译"""
        thread = threading.Thread(target=self.segment_translate)
        thread.daemon = True
        thread.start()

    def build_contextual_segment(self, segment, previous_segment):
        """构建带上下文的片段输入，提升跨句连贯性"""
        if not self.context_mode_var.get() or not previous_segment:
            return segment

        # 仅拼接上一段尾部，控制显存增量
        context = previous_segment[-self.context_overlap_chars:]
        return f"{context}\n{segment}"

    def merge_translation_with_overlap(self, merged_translation, previous_translation):
        """去除因上下文重复导致的翻译前缀重叠"""
        if not previous_translation:
            return merged_translation

        max_overlap = min(len(previous_translation), len(merged_translation), 80)
        overlap_size = 0

        for size in range(max_overlap, 0, -1):
            if previous_translation[-size:] == merged_translation[:size]:
                overlap_size = size
                break

        return merged_translation[overlap_size:]
    
    def segment_translate(self):
        """分段翻译 - 提高长文本翻译质量"""
        try:
            # 获取输入文本
            input_text = self.input_text.get("1.0", tk.END).strip()
            if not input_text:
                messagebox.showwarning("警告", "请输入要翻译的文本！")
                return
            
            # 获取语言代码
            source_selection = self.source_lang.get()
            
            # 自动检测语言
            if source_selection == '自动检测':
                detected_code = self.detect_language(input_text)
                source = detected_code
                
                lang_names = {
                    'zho_Hans': '中文',
                    'eng_Latn': '英语',
                    'jpn_Jpan': '日语',
                    'kor_Hang': '韩语',
                    'fra_Latn': '法语',
                    'deu_Latn': '德语',
                    'spa_Latn': '西班牙语',
                    'rus_Cyrl': '俄语'
                }
                detected_name = lang_names.get(detected_code, '未知')
                
                if detected_code != 'zho_Hans':
                    self.target_lang.set('中文 (zho_Hans)')
                    self.status_bar.config(text=f"检测到{detected_name}，分段翻译为中文...")
            else:
                source = source_selection.split("(")[1].strip(")")
            
            target = self.target_lang.get().split("(")[1].strip(")")
            
            self.translate_button.config(state=tk.DISABLED)
            self.segment_translate_button.config(state=tk.DISABLED, text="翻译中...")
            
            # 分段逻辑：按句号、问号、感叹号分段
            import re
            # 支持中英文标点
            sentences = re.split(r'([.!?。！？\n]+)', input_text)
            
            # 重组句子（保留标点）
            segments = []
            current_segment = ""
            for i in range(0, len(sentences), 2):
                if i < len(sentences):
                    sentence = sentences[i]
                    punctuation = sentences[i+1] if i+1 < len(sentences) else ""
                    current_segment += sentence + punctuation
                    
                    # 每3-5句或超过200字符作为一个片段
                    if len(current_segment) > 200 or punctuation.endswith('\n'):
                        if current_segment.strip():
                            segments.append(current_segment.strip())
                        current_segment = ""
            
            # 添加剩余部分
            if current_segment.strip():
                segments.append(current_segment.strip())
            
            # 如果没有成功分段，按字符数分段
            if len(segments) <= 1 and len(input_text) > 300:
                segments = []
                words = input_text.split()
                current_segment = ""
                for word in words:
                    if len(current_segment) + len(word) > 200:
                        if current_segment.strip():
                            segments.append(current_segment.strip())
                        current_segment = word + " "
                    else:
                        current_segment += word + " "
                if current_segment.strip():
                    segments.append(current_segment.strip())
            
            # 如果还是只有一段，就用普通翻译
            if len(segments) <= 1:
                segments = [input_text]
            
            total_segments = len(segments)
            self.status_bar.config(text=f"分为{total_segments}段，开始翻译...")
            
            # 翻译每个片段（可选上下文连接）
            translated_segments = []
            previous_source_segment = ""
            previous_translation = ""
            for idx, segment in enumerate(segments, 1):
                self.status_bar.config(text=f"正在翻译第{idx}/{total_segments}段...")

                contextual_segment = self.build_contextual_segment(segment, previous_source_segment)
                
                # 编码
                inputs = self.tokenizer(contextual_segment, return_tensors="pt", padding=True)
                
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 翻译
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target),
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
                
                # 解码
                translated_text = self.tokenizer.batch_decode(
                    translated_tokens, skip_special_tokens=True
                )[0]

                if self.context_mode_var.get():
                    translated_text = self.merge_translation_with_overlap(
                        translated_text,
                        previous_translation
                    )
                
                translated_segments.append(translated_text)
                previous_source_segment = segment
                previous_translation = translated_text
                
                # 清理显存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # 合并翻译结果
            final_translation = " ".join(translated_segments)
            
            # 显示结果
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", final_translation)
            self.output_text.config(state=tk.DISABLED)
            
            mode_text = "上下文连接开启" if self.context_mode_var.get() else "标准模式"
            self.status_bar.config(text=f"分段翻译完成！(共{total_segments}段, {mode_text})")
            self.translate_button.config(state=tk.NORMAL)
            self.segment_translate_button.config(state=tk.NORMAL, text="分段翻译")
            
        except torch.cuda.OutOfMemoryError:
            messagebox.showerror("显存不足", "显存不足！翻译失败。\n请尝试翻译更短的文本。")
            self.status_bar.config(text="翻译失败：显存不足")
            self.translate_button.config(state=tk.NORMAL)
            self.segment_translate_button.config(state=tk.NORMAL, text="分段翻译")
            self.warning_label.config(text="⚠️ 显存不足！")
            
        except Exception as e:
            messagebox.showerror("错误", f"翻译失败:\n{str(e)}")
            self.status_bar.config(text="翻译失败")
            self.translate_button.config(state=tk.NORMAL)
            self.segment_translate_button.config(state=tk.NORMAL, text="分段翻译")
            print(f"分段翻译错误: {e}")
    
    def translate(self):
        """执行翻译"""
        try:
            # 获取输入文本
            input_text = self.input_text.get("1.0", tk.END).strip()
            if not input_text:
                messagebox.showwarning("警告", "请输入要翻译的文本！")
                return
            
            # 获取语言代码
            source_selection = self.source_lang.get()
            
            # 自动检测语言
            if source_selection == '自动检测':
                detected_code = self.detect_language(input_text)
                source = detected_code
                
                # 显示检测结果
                lang_names = {
                    'zho_Hans': '中文',
                    'eng_Latn': '英语',
                    'jpn_Jpan': '日语',
                    'kor_Hang': '韩语',
                    'fra_Latn': '法语',
                    'deu_Latn': '德语',
                    'spa_Latn': '西班牙语',
                    'rus_Cyrl': '俄语'
                }
                detected_name = lang_names.get(detected_code, '未知')
                
                # 如果检测到非中文，自动设置目标语言为中文
                if detected_code != 'zho_Hans':
                    self.target_lang.set('中文 (zho_Hans)')
                    self.status_bar.config(text=f"检测到{detected_name}，翻译为中文...")
                else:
                    # 如果是中文，保持用户选择的目标语言
                    self.status_bar.config(text=f"检测到{detected_name}...")
            else:
                # 手动选择的语言
                source = source_selection.split("(")[1].strip(")")
            
            target = self.target_lang.get().split("(")[1].strip(")")
            
            self.translate_button.config(state=tk.DISABLED, text="翻译中...")
            
            # 编码输入文本
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成翻译 - 使用forced_bos_token_id指定目标语言
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target),
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            # 解码
            translated_text = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
            
            # 显示结果
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", translated_text)
            self.output_text.config(state=tk.DISABLED)
            
            if source_selection == '自动检测':
                self.status_bar.config(text=f"翻译完成！(检测: {detected_name} → {self.target_lang.get().split('(')[0].strip()})")
            else:
                self.status_bar.config(text="翻译完成！")
            
            self.translate_button.config(state=tk.NORMAL, text="翻译")
            
        except torch.cuda.OutOfMemoryError:
            messagebox.showerror("显存不足", "显存不足！翻译失败。\n请尝试翻译更短的文本。")
            self.status_bar.config(text="翻译失败：显存不足")
            self.translate_button.config(state=tk.NORMAL, text="翻译")
            self.warning_label.config(text="⚠️ 显存不足！")
            
        except Exception as e:
            messagebox.showerror("错误", f"翻译失败:\n{str(e)}")
            self.status_bar.config(text="翻译失败")
            self.translate_button.config(state=tk.NORMAL, text="翻译")
            print(f"翻译错误详情: {e}")  # 调试用
    
    def clear_text(self):
        """清空文本"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.status_bar.config(text="已清空")

def main():
    root = tk.Tk()
    app = NLLBTranslatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
