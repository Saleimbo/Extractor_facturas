# Streamlit MVP – Extractor de facturas con Azure Document Intelligence
# --------------------------------------------------------------------
# Requisitos de instalación:
#   pip install streamlit pandas azure-ai-documentintelligence python-dotenv \
#               openpyxl xlsxwriter pyyaml python-dateutil pandera
# --------------------------------------------------------------------
# Variables de entorno necesarias:
#   AZURE_DI_ENDPOINT
#   AZURE_DI_KEY
#   AZURE_DI_MODEL_ID         (opcional, por defecto "my-custom-invoice-model")
#   MAPPER_YAML_PATH          (opcional, YAML de columnas)
#   CLEANING_YAML_PATH        (opcional, YAML de reglas de limpieza)
# --------------------------------------------------------------------

import os
import io
import re
import time
import zipfile
import hashlib
from decimal import Decimal, InvalidOperation
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import base64
import yaml
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from dateutil import parser as date_parser
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

load_dotenv()

###############################################################################
# Helper – Azure DI client
###############################################################################

def _load_client() -> DocumentIntelligenceClient:
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
    if not endpoint or not key:
        raise RuntimeError("Faltan credenciales de Azure.")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

###############################################################################
# YAML loaders
###############################################################################

def _safe_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _load_mapping() -> Dict[str, str]:
    return _safe_yaml(os.getenv("MAPPER_YAML_PATH"))

def _load_cleaning() -> Dict[str, Any]:
    data = _safe_yaml(os.getenv("CLEANING_YAML_PATH"))
    return data.get("campos", {}) if data else {}

###############################################################################
# Normalisation utilities
###############################################################################

def _clean_date(value: str, fmt: str) -> Tuple[str, bool]:
    try:
        dt = date_parser.parse(str(value), dayfirst=True, fuzzy=True)
        return dt.strftime(fmt), True
    except (ValueError, TypeError):
        return value, False

def _clean_amount(value: str) -> Tuple[Any, bool]:
    try:
        clean = re.sub(r"[€%\s]", "", str(value))
        clean = clean.replace(".", "").replace(",", ".")
        amount = Decimal(clean)
        return float(amount), True
    except (InvalidOperation, ValueError):
        return value, False

def _clean_nif(value: str, strip_prefix: List[str] = None, keep_country_code: bool = True) -> Tuple[str, bool]:
    if not value:
        return value, False
    cleaned = str(value).upper()
    for prefix in strip_prefix or []:
        cleaned = cleaned.replace(prefix.upper(), "")
    if keep_country_code:
        match = re.match(r"^([A-Z]{2})([A-Z0-9]{8,})$", cleaned)
        if match:
            return f"{match.group(1)}{match.group(2)}", True
    return cleaned, bool(re.fullmatch(r"[A-Z0-9]{8,}", cleaned))

def _clean_entero(value: str, remove_symbols: List[str] = None) -> Tuple[int, bool]:
    try:
        clean = str(value)
        for symbol in remove_symbols or []:
            clean = clean.replace(symbol, "")
        numero = re.search(r"\d+", clean).group()
        return int(numero), True
    except (TypeError, AttributeError):
        return value, False

def _clean_name(value: str) -> Tuple[str, bool]:
    if value is None:
        return value, False
    s = str(value).strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        cleaned = f"{parts[1]} {parts[0]}".strip()
        return cleaned, True
    return s, True

def _create_file_link(filename: str, content: bytes) -> str:
    """Devuelve una data‑URL (PDF o imagen) para abrir en otra pestaña."""
    b64 = base64.b64encode(content).decode()
    ext = filename.split(".")[-1].lower()
    mime = "application/pdf" if ext == "pdf" else f"image/{ext}"
    return f"data:{mime};base64,{b64}"

_CLEAN_FUNCS = {
    "fecha": _clean_date,
    "importe": _clean_amount,
    "nif": _clean_nif,
    "nombre_prop": _clean_name,
    "entero": _clean_entero,
}

###############################################################################
# Document processing
###############################################################################

@st.cache_data(show_spinner=False)
def _analyze_document(file_hash: str, content: bytes, model_id: str) -> Dict[str, Any]:
    client = _load_client()
    poller = client.begin_analyze_document(model_id=model_id, body=content)
    result: AnalyzeResult = poller.result()
    if not result.documents:
        return {"_error": "No se detectó documento"}
    doc = result.documents[0]
    row: Dict[str, Any] = {}
    conf: Dict[str, float] = {}
    for name, field in doc.fields.items():
        if field:
            row[name] = field.content or ""
            conf[name] = getattr(field, "confidence", 1.0)
        else:
            row[name] = ""
            conf[name] = 0.0
    row["_conf"] = conf
    return row

###############################################################################
# Cleaning pipeline
###############################################################################

def _clean_row(row: Dict[str, Any], cleaning_rules: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    cleaned: Dict[str, Any] = {}
    flags: Dict[str, str] = {}
    conf = row.pop("_conf", {})
    for key, value in row.items():
        rule = cleaning_rules.get(key, {})
        tipo = rule.get("tipo")
        ok_conf = conf.get(key, 1.0) >= 0.8
        cleaned_val = value
        success = True
        if tipo in _CLEAN_FUNCS:
            if tipo == "fecha":
                fmt = rule.get("formato", "%d/%m/%Y")
                cleaned_val, success = _clean_date(value, fmt)
            elif tipo == "importe":
                cleaned_val, success = _clean_amount(value)
            elif tipo == "nif":
                strip_prefix = rule.get("strip_prefix", [])
                cleaned_val, success = _clean_nif(value, strip_prefix)
            elif tipo == "nombre_prop":
                cleaned_val, success = _clean_name(value)
            elif tipo == "entero":
                remove_symbols = rule.get("remove_symbols", [])
                cleaned_val, success = _clean_entero(value, remove_symbols)
        cleaned[key] = cleaned_val
        if not ok_conf:
            flags[key] = "ocr_low"
        elif not success:
            flags[key] = "clean_fail"
    return cleaned, flags

###############################################################################
# Schema validation using Pandera
###############################################################################

def _build_schema(cleaning_rules: Dict[str, Any]) -> DataFrameSchema:
    columns = {}
    for col, rule in cleaning_rules.items():
        tipo = rule.get("tipo")
        if tipo == "importe":
            columns[col] = Column(float, Check.gt(0))
        elif tipo == "fecha":
            columns[col] = Column(object)  # será str formateado
        else:
            columns[col] = Column(object)
    return DataFrameSchema(columns)

###############################################################################
# File helpers
###############################################################################

def _extract_docs(uploaded_files) -> List[Tuple[str, bytes]]:
    docs: List[Tuple[str, bytes]] = []
    for f in uploaded_files:
        if f.type == "application/zip":
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    if info.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
                        docs.append((info.filename, z.read(info)))
        else:
            docs.append((f.name, f.read()))
    return docs

###############################################################################
# Main Streamlit app
###############################################################################

def main():
    st.set_page_config(page_title="Extractor de facturas", page_icon="🧾", layout="wide")
    st.title("🧾 Extractor de facturas Demo")

    mapping = _load_mapping()
    cleaning_rules = _load_cleaning()

    uploaded_files = st.file_uploader(
        "Arrastra PDFs, imágenes o ZIPs",
        type=["pdf", "jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Sube al menos un archivo para comenzar.")
        return

    docs = _extract_docs(uploaded_files)
    total = len(docs)
    if total == 0:
        st.warning("El ZIP no contenía archivos válidos.")
        return

    model_id = os.getenv("AZURE_DI_MODEL_ID")

    rows: List[Dict[str, Any]] = []
    flags_matrix: List[Dict[str, str]] = []

    t0 = time.time()
    progress = st.progress(0.0, text="Procesando documentos…")

    def _process(item):
        filename, content = item
        file_hash = hashlib.sha1(content).hexdigest()
        raw = _analyze_document(file_hash, content, model_id)
        raw["_filename"] = filename
        cleaned, flags = _clean_row(raw, cleaning_rules)
        cleaned["_filename"] = filename
        return cleaned, flags

    with ThreadPoolExecutor(max_workers=4) as pool:
        for idx, (cleaned, flags) in enumerate(pool.map(_process, docs), start=1):
            rows.append(cleaned)
            flags_matrix.append(flags)
            progress.progress(idx / total, text=f"Procesado {idx}/{total}")

    df = pd.DataFrame(rows)

    # ----------------------------------------------------------
    # Columna con enlace (data‑URL) que abre el documento
    # ----------------------------------------------------------
    df["Documento"] = [_create_file_link(fn, ct) for fn, ct in docs]

    flags_df = pd.DataFrame(flags_matrix).reindex(columns=df.columns, fill_value=None)

    # Apply mapping
    if mapping:
        df = df.rename(columns=mapping)
        flags_df = flags_df.rename(columns=mapping)
        ordered = list(mapping.values())
        other = [c for c in df.columns if c not in ordered]
        df = df[ordered + other]
        flags_df = flags_df[ordered + other]

    # Schema validation
    try:
        schema = _build_schema(cleaning_rules)
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for _, fc in exc.failure_cases.iterrows():
            idx_orig = fc["index"]
            col_orig = fc["column"]
            col_final = mapping.get(col_orig, col_orig)
            if col_final in flags_df.columns and idx_orig in flags_df.index:
                flags_df.at[idx_orig, col_final] = "clean_fail"

    ###############################################################################
    # Regla de coherencia: subtotal + IVA = total
    ###############################################################################
    # Nombres tal y como aparecen YA renombrados en df y flags_df
    COL_SUBTOTAL = "base_imponible"   # ó mapping.get("subtotal", "Subtotal")
    COL_IVA      = "importe_iva"
    COL_TOTAL    = "total"

    tolerance = 0.01  # 1 céntimo de margen

    if all(c in df.columns for c in (COL_SUBTOTAL, COL_IVA, COL_TOTAL)):
        # Calculamos la diferencia absoluta
        diff = (df[COL_SUBTOTAL].fillna(0).astype(float)
                + df[COL_IVA].fillna(0).astype(float)
                - df[COL_TOTAL].fillna(0).astype(float)).abs()

        # Filas donde la diferencia supera la tolerancia
        mismatch_mask = diff > tolerance

        # Marca la celda ‘Total’ con un flag especial
        flags_df.loc[mismatch_mask, COL_TOTAL] = "sum_mismatch"


    ###############################################################################
    # Estilos de celdas según los flags
    ###############################################################################
    ERROR_COLOR = "#FFD6D6"  # rojo suave

    cell_style = {}
    for row_idx, row_flags in flags_df.iterrows():
        for col, flag in row_flags.items():
            if flag:  # ‘ocr_low’, ‘clean_fail’, ‘sum_mismatch’, etc.
                cell_style[(row_idx, col)] = {"backgroundColor": ERROR_COLOR}


    t1 = time.time()
    avg = (t1 - t0) / total
    st.metric("⏱️ Tiempo medio", f"{avg:0.2f} s/factura")

    # -----------------------------------------------------------------
    # Tabla editable sencilla con st.data_editor + LinkColumn
    # -----------------------------------------------------------------
    st.subheader("Revisa y edita los datos extraídos")

    edited_df = st.data_editor(
        df,
        column_config={
            "Documento": st.column_config.LinkColumn(
                label="Abrir documento",
                display_text="Ver",
            )
        },
        disabled=["Documento", "_filename"],   # estas columnas no se editan
        num_rows="dynamic",
        use_container_width=True,
    )

    
    # Recalcula incoherencias por si el usuario cambió algo
    tol = 0.01
    mismatch = (edited_df["base_imponible"] + edited_df["importe_iva"] 
                - edited_df["total"]).abs() > tol
    

    cols_to_hide = ["Documento", "_filename"]          # añade aquí otras que no quieras mostrar
    view_df = edited_df.drop(columns=cols_to_hide, errors="ignore").copy()
    
    importe_cols = ["base_imponible", "importe_iva", "total"]
    view_df[importe_cols] = view_df[importe_cols].applymap(
        lambda x: f"{x:,.2f}".replace(",", " ").replace(".", ",")  # 309,08 → 309,08
    )

    def highlight_total(col):
        return ["background-color: #FFD6D6" if mismatch.loc[idx] else 
                "" for idx in col.index]
    
    
    st.markdown("#### Incoherencias detectadas")
    st.dataframe(
        
        view_df.style.apply(highlight_total, subset=["total"]),
        use_container_width=True,
    )

    

    # -----------------------------------------------------------------
    # Exportar a CSV / Excel / ZIP
    # -----------------------------------------------------------------
    st.markdown("### Exportar")

    csv_df = edited_df.drop(columns=["_filename", "Documento"], errors="ignore")
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️  Descargar CSV", csv_bytes, "facturas.csv", "text/csv")

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        csv_df.to_excel(writer, index=False, sheet_name="Facturas")
    excel_buf.seek(0)
    st.download_button(
        "⬇️  Descargar Excel",
        excel_buf.getvalue(),
        "facturas.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("facturas.csv", csv_bytes)
        zf.writestr("facturas.xlsx", excel_buf.getvalue())
    zip_buf.seek(0)
    st.download_button("⬇️  Descargar ZIP", zip_buf.getvalue(), "facturas.zip", "application/zip")


if __name__ == "__main__":
    main()
