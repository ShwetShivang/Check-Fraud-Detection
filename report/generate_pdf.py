"""
Generate PDF report from the project results.
Uses fpdf2 to create a professional report.
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from fpdf import FPDF


class ReportPDF(FPDF):
    """Custom PDF with header/footer."""

    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Cheque Fraud Detection using Ink Variation Analysis',
                  new_x='LMARGIN', new_y='NEXT', align='C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(20, 60, 120)
        self.ln(6)
        self.cell(0, 12, title, new_x='LMARGIN', new_y='NEXT', align='L')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(40, 80, 140)
        self.ln(4)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT', align='L')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, f'  -  {text}')
        self.ln(1)

    def add_image_safe(self, path, w=170):
        """Add image if it exists."""
        if os.path.exists(path):
            try:
                self.image(path, x=20, w=w)
                self.ln(6)
                return True
            except Exception as e:
                self.body_text(f"[Image not available: {os.path.basename(path)}]")
                return False
        return False


def generate_report():
    """Generate the full PDF report."""
    print("[Report] Generating PDF report...")

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ─── Title Page ───────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, 'Cheque Fraud Detection', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 12, 'using Ink Variation Analysis', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(20)

    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, 'AI-Powered Fraud Detection System', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(10)

    pdf.set_font('Helvetica', 'I', 12)
    pdf.cell(0, 8, 'Patch-based Ink Similarity Learning', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.cell(0, 8, 'Siamese Network | CNN | Classical ML', new_x='LMARGIN', new_y='NEXT', align='C')

    pdf.ln(30)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, 'Based on: Dansena, Bag & Pal - ICPRMI 2017', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.cell(0, 7, 'Dataset: IDRBT Cheque Data (112 images)', new_x='LMARGIN', new_y='NEXT', align='C')

    # ─── Abstract ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title('Abstract')
    pdf.body_text(
        'This project presents an AI-based system for detecting fraudulent modifications '
        'in bank cheque images by analyzing pen ink variations. When a fraudster alters a '
        'cheque by adding or modifying text, the added ink -- even if visually similar -- '
        'often differs from the original ink at a spectral level. Our system leverages this '
        'principle using a patch-based comparison approach rather than traditional pixel-level '
        'segmentation, making it suitable for small datasets without segmentation masks.'
    )
    pdf.body_text(
        'We implement and compare three distinct approaches: (1) a Siamese Network with '
        'contrastive loss for ink similarity learning, (2) a CNN baseline classifier for '
        'direct ink class prediction, and (3) a Classical Machine Learning pipeline using '
        'hand-crafted features (GLCM, LBP, LAB color histograms) with SVM and Random Forest '
        'classifiers. A Streamlit web interface provides accessible real-time cheque analysis '
        'with Grad-CAM explainability visualizations.'
    )

    # ─── Introduction ─────────────────────────────────────────────
    pdf.chapter_title('1. Introduction')

    pdf.section_title('1.1 Background')
    pdf.body_text(
        'Bank cheque fraud remains a significant financial crime worldwide. One common '
        'technique involves adding or altering text on a legitimate cheque using a different '
        'pen with visually similar ink. While such modifications may fool the human eye, '
        'computational analysis of ink properties can reveal subtle differences that indicate '
        'tampering.'
    )
    pdf.body_text(
        'Traditional forensic methods for ink analysis include chemical testing and '
        'spectroscopic analysis, both of which are destructive and expensive. Non-destructive, '
        'image-based approaches offer a compelling alternative, leveraging digital image '
        'processing and machine learning to detect ink inconsistencies from scanned images alone.'
    )

    pdf.section_title('1.2 Problem Statement')
    pdf.body_text(
        'Given a scanned cheque image potentially written with multiple pens:'
    )
    pdf.bullet_point('Determine whether the cheque has been tampered with')
    pdf.bullet_point('Localize the regions where ink differences occur')
    pdf.bullet_point('Quantify the confidence of the fraud detection')

    pdf.section_title('1.3 Key Challenges')
    pdf.bullet_point('Small Dataset: Only 112 cheque images available for training')
    pdf.bullet_point('No Pixel-Level Labels: No segmentation masks or ink boundary annotations')
    pdf.bullet_point('Visually Similar Inks: Different pens may produce nearly identical colors')
    pdf.bullet_point('Varying Handwriting: Different writers, speeds, and pen pressures')

    # ─── Literature Review ────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title('2. Literature Review')

    pdf.section_title('2.1 Ink Differentiation in Documents')
    pdf.body_text(
        'The foundational work by Dansena, Bag, and Pal (2017) on "Differentiating Pen Inks '
        'in Hand-written Bank Cheques Using Multi-Layer Perceptron" established a non-destructive '
        'approach to ink analysis. Their method extracted 15 statistical features from ink pixels '
        'across color channels, used K-means clustering for ink pixel identification, and achieved '
        '94.6% accuracy for known pen samples and 93.5% for unknown pens.'
    )

    pdf.section_title('2.2 Siamese Networks')
    pdf.body_text(
        'Siamese networks (Bromley et al., 1994; Koch et al., 2015) excel at few-shot and '
        'similarity learning tasks. Their shared-weight architecture naturally handles small '
        'datasets, unseen classes, and binary comparison tasks.'
    )

    pdf.section_title('2.3 LAB Color Space')
    pdf.body_text(
        'The CIELAB color space separates luminance (L) from chromaticity (A, B), making it '
        'particularly suitable for ink analysis. The L channel captures ink density, while A and '
        'B channels capture subtle color differences between ink formulations.'
    )

    # ─── Methodology ──────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title('3. Methodology')

    pdf.section_title('3.1 Dataset')
    pdf.body_text(
        'IDRBT Cheque Dataset (via Kaggle): 112 cheque images, each written with two different '
        'pens. 9 different writers used 14 distinct pens, yielding 126 pen-writer combinations. '
        'A synthetic data generator supplements the dataset for development and testing.'
    )

    pdf.section_title('3.2 Data Preprocessing')
    pdf.body_text(
        'Images are resized to 256x512 pixels and normalized to [0,1]. Color space conversions '
        'include RGB to Grayscale (for texture) and RGB to LAB (for ink analysis). Adaptive '
        'thresholding extracts ink masks. 64x64 patches are generated via sliding window with '
        'stride=32. Augmentation includes rotation (+/-15 deg), Gaussian noise, and brightness variation.'
    )

    pdf.section_title('3.3 Model Architectures')

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'A. Siamese Network (Primary Model, ~200K parameters)', new_x='LMARGIN', new_y='NEXT')
    pdf.body_text(
        'Twin CNN branches with shared weights. Architecture: Conv2d(3->32) -> BN -> ReLU -> '
        'MaxPool -> Conv2d(32->64) -> BN -> ReLU -> MaxPool -> Conv2d(64->128) -> BN -> ReLU -> '
        'AdaptiveAvgPool -> FC(64). Trained with Contrastive Loss using Euclidean distance.'
    )

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'B. CNN Baseline (~100K parameters)', new_x='LMARGIN', new_y='NEXT')
    pdf.body_text(
        'Standard CNN classifier: Conv2d(3->16) -> Conv2d(16->32) -> Conv2d(32->64) -> '
        'AdaptiveAvgPool -> FC(32) -> FC(2). Trained with Cross-Entropy loss.'
    )

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'C. Classical ML', new_x='LMARGIN', new_y='NEXT')
    pdf.body_text(
        'Feature extraction: GLCM texture features (contrast, dissimilarity, homogeneity, '
        'energy, correlation), LBP histogram (10 bins), LAB color histogram (3x32 bins). '
        'Combined ~146-dimensional feature vector. Classifiers: SVM (RBF kernel) and '
        'Random Forest (100 trees).'
    )

    pdf.section_title('3.4 Fraud Detection Pipeline')
    pdf.body_text(
        'For each cheque: (1) Preprocess and extract patches, (2) Identify left/right pen '
        'regions, (3) Compare cross-region patch pairs using Siamese model, (4) Compute '
        'similarity scores and heatmap, (5) If >20% patches have low similarity -> FRAUD.'
    )

    # ─── Results ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title('4. Results')

    pdf.section_title('4.1 Model Comparison')

    # Results table
    pdf.set_font('Helvetica', 'B', 11)
    col_w = [45, 30, 30, 30, 30]
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, 1, 0, 'C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 11)
    results = [
        ['Siamese', '0.7317', '0.6936', '0.8300', '0.7557'],
        ['CNN Baseline', '0.6314', '0.5775', '0.6989', '0.6324'],
        ['Classical SVM', '0.7242', '0.6493', '0.8059', '0.7192'],
        ['Classical RF', '0.7758', '0.7456', '0.7412', '0.7434'],
    ]
    for row in results:
        for i, cell in enumerate(row):
            pdf.cell(col_w[i], 7, cell, 1, 0, 'C')
        pdf.ln()

    pdf.ln(6)
    pdf.body_text(
        'Note: Results obtained using synthetic data (50 cheque images). Performance on the '
        'real IDRBT dataset (112 images) may vary based on ink diversity and image quality.'
    )

    # Add figures if they exist
    pdf.section_title('4.2 Visualizations')

    figures = [
        ('model_comparison.png', 'Model Performance Comparison'),
        ('roc_curves.png', 'ROC Curves'),
        ('siamese_training.png', 'Siamese Network Training Curves'),
        ('siamese_confusion.png', 'Siamese Confusion Matrix'),
        ('siamese_distance_dist.png', 'Siamese Distance Distribution'),
        ('cnn_training.png', 'CNN Training Curves'),
        ('cnn_confusion.png', 'CNN Confusion Matrix'),
        ('classical_svm_confusion.png', 'SVM Confusion Matrix'),
        ('classical_rf_confusion.png', 'RF Confusion Matrix'),
    ]

    for fname, caption in figures:
        fpath = os.path.join(config.FIGURES_DIR, fname)
        if os.path.exists(fpath):
            # Check if we need a new page
            if pdf.get_y() > 200:
                pdf.add_page()
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6, f'Figure: {caption}', 0, 1, 'C')
            pdf.add_image_safe(fpath, w=160)
            pdf.ln(4)

    # ─── Conclusion ───────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title('5. Conclusion')
    pdf.body_text(
        'This project demonstrates the feasibility of automated cheque fraud detection '
        'through ink variation analysis. Key contributions include:'
    )
    pdf.bullet_point(
        'Patch-based formulation: Overcomes the lack of pixel-level labels by '
        'framing the problem as patch comparison'
    )
    pdf.bullet_point(
        'Siamese Network approach: Provides robust performance on small datasets '
        'through metric learning'
    )
    pdf.bullet_point(
        'Multi-model comparison: Demonstrates trade-offs between deep learning '
        'and classical approaches'
    )
    pdf.bullet_point(
        'Complete pipeline: End-to-end system from image input to fraud verdict '
        'with visual explanations'
    )
    pdf.bullet_point(
        'Accessible interface: Streamlit application for practical deployment'
    )

    # ─── Future Work ──────────────────────────────────────────────
    pdf.chapter_title('6. Future Work')
    pdf.bullet_point(
        'U-Net Segmentation: With pixel-level ink labels, a U-Net architecture could '
        'precisely segment different ink regions'
    )
    pdf.bullet_point(
        'Vision Transformers: Self-attention could capture long-range ink dependencies'
    )
    pdf.bullet_point(
        'Larger Datasets: Scaling to thousands of cheque images would improve generalization'
    )
    pdf.bullet_point(
        'Multi-spectral Imaging: Near-infrared and UV imaging could enhance ink differentiation'
    )
    pdf.bullet_point(
        'Transfer Learning: Pre-trained models (ResNet, EfficientNet) with fine-tuning'
    )

    # ─── References ───────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title('7. References')

    refs = [
        '[1] P. Dansena, S. Bag, and R. Pal, "Differentiating Pen Inks in Hand-written '
        'Bank Cheques Using Multi-Layer Perceptron", Proc. of 7th ICPRMI, Kolkata, India, '
        'December 2017.',

        '[2] J. Bromley, I. Guyon, Y. LeCun, E. Sackinger, and R. Shah, "Signature '
        'Verification Using a Siamese Time Delay Neural Network", NIPS, 1994.',

        '[3] G. Koch, R. Zemel, and R. Salakhutdinov, "Siamese Neural Networks for '
        'One-shot Image Recognition", ICML Deep Learning Workshop, 2015.',

        '[4] R.M. Haralick, K. Shanmugam, and I. Dinstein, "Textural Features for '
        'Image Classification", IEEE Trans. SMC, 1973.',

        '[5] T. Ojala, M. Pietikainen, and D. Harwood, "A Comparative Study of Texture '
        'Measures with Classification Based on Feature Distributions", Pattern Recognition, 1996.',

        '[6] R.R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks '
        'via Gradient-based Localization", ICCV 2017.',
    ]

    for ref in refs:
        pdf.body_text(ref)

    # ─── Save ─────────────────────────────────────────────────────
    output_path = os.path.join(config.BASE_DIR, 'report', 'Cheque_Fraud_Detection_Report.pdf')
    pdf.output(output_path)
    print(f"[Report] PDF saved: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()
