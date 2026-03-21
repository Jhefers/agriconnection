from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse
import html
import json
import math as _math_mod

import pandas as pd
from pandas.api.types import is_numeric_dtype


BASE_DIR = Path(__file__).resolve().parent


# CDs válidos para composição de rota HUB.
HUBS_PADRAO = ["BARUERI", "ITU", "EMBU"]


@dataclass(frozen=True)
class ColunasPadrao:
    operacao: str = "OPERAÇÃO 1"
    destino: str = "DESTINO"
    uf_destino: str = "UF -DESTINO"
    tipo_transp: str = "TIPO TRANSP"
    transportadora: str = "TRANSPORTADORA"
    veiculo_kg: str = "VEICULO KG"
    frete: str = "FRETE"
    origem: str = "ORIGEM"


def normalizar_nome_coluna(col: str) -> str:
    return " ".join(str(col).replace("\n", " ").replace("\r", " ").strip().split()).upper()


def mapear_colunas(df: pd.DataFrame, obrigatorias: Iterable[str]) -> Dict[str, str]:
    nome_original_por_normalizado = {normalizar_nome_coluna(c): c for c in df.columns}
    mapeamento: Dict[str, str] = {}

    for col_esperada in obrigatorias:
        col_norm = normalizar_nome_coluna(col_esperada)
        if col_norm not in nome_original_por_normalizado:
            disponiveis = ", ".join([str(c) for c in df.columns])
            raise ValueError(
                f"Coluna obrigatória não encontrada: '{col_esperada}'. Colunas disponíveis: {disponiveis}"
            )
        mapeamento[col_esperada] = nome_original_por_normalizado[col_norm]
    return mapeamento


def _tem_colunas_esperadas(df: pd.DataFrame, obrigatorias: Iterable[str]) -> bool:
    cols_norm = {normalizar_nome_coluna(c) for c in df.columns}
    obrig_norm = {normalizar_nome_coluna(c) for c in obrigatorias}
    return obrig_norm.issubset(cols_norm)


def _detectar_header_excel(path: Path, obrigatorias: Iterable[str], max_linhas: int = 30) -> int | None:
    bruto = pd.read_excel(path, header=None, nrows=max_linhas)
    obrig_norm = {normalizar_nome_coluna(c) for c in obrigatorias}

    melhor_linha = None
    melhor_score = 0

    for idx in range(len(bruto)):
        row_values = [v for v in bruto.iloc[idx].tolist() if pd.notna(v)]
        row_norm = {normalizar_nome_coluna(v) for v in row_values}
        score = len(obrig_norm.intersection(row_norm))
        if score > melhor_score:
            melhor_score = score
            melhor_linha = idx

    # Exige ao menos 2 colunas-chave para evitar falso positivo.
    if melhor_score >= 2:
        return melhor_linha
    return None


def ler_tabela(path: Path, obrigatorias: Iterable[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        if obrigatorias and not _tem_colunas_esperadas(df, obrigatorias):
            header_row = _detectar_header_excel(path, obrigatorias)
            if header_row is not None:
                df = pd.read_excel(path, header=header_row)
        return df
    if suffix == ".csv":
        # Tenta ';' primeiro (padrão comum em PT-BR) e cai para ','.
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            df = pd.read_csv(path)

        if obrigatorias and not _tem_colunas_esperadas(df, obrigatorias):
            raise ValueError(
                "CSV lido sem colunas esperadas. Verifique se o cabeçalho está presente e o separador está correto."
            )
        return df
    raise ValueError(f"Formato não suportado: {path}")


def resolver_caminho(path: Path) -> Path:
    if path.is_absolute():
        return path
    return BASE_DIR / path


def normalizar_veiculo(valor: object) -> str | None:
    if pd.isna(valor):
        return None

    texto = str(valor).strip().upper()
    if not texto or texto == "NAN":
        return None

    numero = pd.to_numeric(texto, errors="coerce")
    if pd.notna(numero):
        numero_float = float(numero)
        if numero_float.is_integer():
            return str(int(numero_float))
        return str(numero_float)

    return texto


def chave_ordenacao_veiculo(valor: object) -> tuple[int, float | str]:
    texto = "" if valor is None else str(valor).strip().upper()
    numero = pd.to_numeric(texto, errors="coerce")
    if pd.notna(numero):
        return (0, float(numero))
    return (1, texto)


def preparar_numerico_veiculo(df: pd.DataFrame, col_veiculo: str, col_frete: str) -> pd.DataFrame:
    out = df.copy()
    out[col_veiculo] = out[col_veiculo].apply(normalizar_veiculo)

    if is_numeric_dtype(out[col_frete]):
        out[col_frete] = pd.to_numeric(out[col_frete], errors="coerce")
    else:
        frete_str = out[col_frete].astype(str).str.replace("R$", "", regex=False).str.strip()

        # Quando há vírgula, assume formato PT-BR. O ponto passa a ser milhar.
        mask_com_virgula = frete_str.str.contains(",", regex=False, na=False)
        frete_str.loc[mask_com_virgula] = (
            frete_str.loc[mask_com_virgula]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        # Sem vírgula, preserva o ponto como decimal nativo do Excel/exportação.
        frete_str.loc[~mask_com_virgula] = frete_str.loc[~mask_com_virgula].str.replace(" ", "", regex=False)

        out[col_frete] = pd.to_numeric(frete_str, errors="coerce")

    out = out.dropna(subset=[col_veiculo, col_frete])
    return out


def padronizar_texto(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(str).str.strip().str.upper()
    return out


def gerar_tabela_consolidada(
    inbound: pd.DataFrame,
    transferencia: pd.DataFrame,
    hubs: List[str],
    cols: ColunasPadrao,
) -> pd.DataFrame:
    required_inbound = [
        cols.operacao,
        cols.destino,
        cols.uf_destino,
        cols.tipo_transp,
        cols.transportadora,
        cols.veiculo_kg,
        cols.frete,
    ]
    required_transf = [
        cols.operacao,
        cols.origem,
        cols.destino,
        cols.uf_destino,
        cols.tipo_transp,
        cols.transportadora,
        cols.veiculo_kg,
        cols.frete,
    ]

    map_in = mapear_colunas(inbound, required_inbound)
    map_tr = mapear_colunas(transferencia, required_transf)

    inb = inbound.rename(columns={v: k for k, v in map_in.items()})
    trf = transferencia.rename(columns={v: k for k, v in map_tr.items()})

    inb = padronizar_texto(inb, [cols.destino, cols.transportadora, cols.tipo_transp, cols.operacao, cols.uf_destino])
    trf = padronizar_texto(
        trf,
        [cols.origem, cols.destino, cols.transportadora, cols.tipo_transp, cols.operacao, cols.uf_destino],
    )

    inb = preparar_numerico_veiculo(inb, cols.veiculo_kg, cols.frete)
    trf = preparar_numerico_veiculo(trf, cols.veiculo_kg, cols.frete)

    # ROTAS DIRETAS: Porto -> Destino final (usa apenas inbound).
    diretas = pd.DataFrame(
        {
            "DESTINO": inb[cols.destino],
            "ROTA": "DIRETO",
            "HUB": "",
            "TRANSP. 1": inb[cols.transportadora],
        "VEICULO 1": inb[cols.veiculo_kg],
            "CUSTO 1": inb[cols.frete],
            "TRANSP. 2": "",
            "VEICULO 2": "",
            "CUSTO 2": 0.0,
        }
    )

    # ROTAS HUB: Porto -> CD (inbound) + CD -> destino final (transferência).
    hubs_norm = [h.strip().upper() for h in hubs]
    inb_hub = inb[inb[cols.destino].isin(hubs_norm)].copy()
    trf_hub = trf[trf[cols.origem].isin(hubs_norm)].copy()

    hub_join = inb_hub.merge(
        trf_hub,
        left_on=[cols.destino, cols.veiculo_kg],
        right_on=[cols.origem, cols.veiculo_kg],
        how="inner",
        suffixes=("_IN", "_TR"),
    )

    hubs_df = pd.DataFrame(
        {
            "DESTINO": hub_join[f"{cols.destino}_TR"],
            "ROTA": "HUB",
        "HUB": hub_join[f"{cols.destino}_IN"],
            "TRANSP. 1": hub_join[f"{cols.transportadora}_IN"],
        "VEICULO 1": hub_join[cols.veiculo_kg],
            "CUSTO 1": hub_join[f"{cols.frete}_IN"],
            "TRANSP. 2": hub_join[f"{cols.transportadora}_TR"],
        "VEICULO 2": hub_join[cols.veiculo_kg],
            "CUSTO 2": hub_join[f"{cols.frete}_TR"],
        }
    )

    resultado = pd.concat([diretas, hubs_df], ignore_index=True)
    resultado["TOTAL"] = (resultado["CUSTO 1"] + resultado["CUSTO 2"]).round(2)
    menor_total = resultado.groupby(["DESTINO", "VEICULO 1"])["TOTAL"].transform("min").round(2)
    resultado["MELHOR_POR_DESTINO_VEICULO"] = resultado["TOTAL"].round(2).eq(menor_total)

    resultado = resultado.sort_values(["DESTINO", "TOTAL", "ROTA", "HUB"]).reset_index(drop=True)
    return resultado


def _formatar_brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _options_html(values: Iterable[str], include_all: bool = True) -> str:
    opts = []
    if include_all:
        opts.append('<option value="">Todos</option>')
    for v in values:
        escaped = html.escape(str(v))
        opts.append(f'<option value="{escaped}">{escaped}</option>')
    return "\n".join(opts)


def gerar_html(df: pd.DataFrame, destino_saida: Path) -> None:
    destinos = sorted(df["DESTINO"].astype(str).unique().tolist())

    produtos_data: List[dict] = []
    capacidades_data: List[dict] = []

    path_prod = destino_saida.parent / "Parametros_Produtos.xlsx"
    if path_prod.exists():
        try:
            df_prod = carregar_parametros_produtos(path_prod)
            if {"PRODUTO", "PESO LIQUIDO", "QUANTIDADE POR PALLET"}.issubset(df_prod.columns):
                for _, row in df_prod.iterrows():
                    produtos_data.append(
                        {
                            "codigo": str(row.get("CODIGO", "")).strip(),
                            "produto": str(row.get("PRODUTO", "")).strip(),
                            "peso_unitario": float(row.get("PESO LIQUIDO", 0) or 0),
                          "fator_peso_bruto": float(row.get("FATOR PESO BRUTO", 1) or 1),
                            "qtd_por_pallet": float(row.get("QUANTIDADE POR PALLET", 1) or 1),
                        }
                    )
        except Exception:
            produtos_data = []

    path_cap = destino_saida.parent / "Capacidade_veiculo_por_Transportadora.xlsx"
    if path_cap.exists():
        try:
            df_cap = carregar_capacidade_veiculos(path_cap)
            if {"TIPO VEICULO", "PESO MAXIMO POR VEICULO", "PALETES", "OPERADOR"}.issubset(df_cap.columns):
                for _, row in df_cap.iterrows():
                    paletes = row.get("PALETES")
                    capacidades_data.append(
                        {
                            "tipo": str(row.get("TIPO VEICULO", "")).strip().upper(),
                            "peso_max": float(row.get("PESO MAXIMO POR VEICULO", 0) or 0),
                            "paletes": None if pd.isna(paletes) else int(paletes),
                            "operador": str(row.get("OPERADOR", "")).strip().upper(),
                        }
                    )
        except Exception:
            capacidades_data = []

    rotas_data: List[dict] = []
    for _, row in df.iterrows():
        rotas_data.append(
            {
                "origem": "PORTO STS",
                "destino": str(row["DESTINO"]).strip().upper(),
                "rota": str(row["ROTA"]).strip().upper(),
                "hub": str(row["HUB"]).strip().upper(),
                "transp1": str(row["TRANSP. 1"]).strip().upper(),
              "transp2": str(row["TRANSP. 2"]).strip().upper(),
                "veiculo1": str(row["VEICULO 1"]).strip().upper(),
              "veiculo2": str(row["VEICULO 2"]).strip().upper(),
              "custo1": float(row["CUSTO 1"]),
              "custo2": float(row["CUSTO 2"]),
                "total": float(row["TOTAL"]),
            }
        )

    produtos_unicos = sorted(
        [p["produto"] for p in produtos_data if p.get("produto")],
        key=lambda x: x.upper(),
    )
    produtos_options = _options_html(produtos_unicos, include_all=False) if produtos_unicos else ""

    simulador_ativo = bool(produtos_data and capacidades_data)
    simulador_msg = (
      "Informe os parâmetros da carga para gerar as opções operacionais mais competitivas."
        if simulador_ativo
      else "Simulador indisponível: inclua Parametros_Produtos.xlsx e Capacidade_veiculo_por_Transportadora.xlsx."
    )

    logo_candidates = [
      "logo-agri-dark.png",
      "logo-agri-dark_old.svg",
      "logo-agri-dark.svg",
      "logo.svg",
      "logo.png",
      "logo.jpg",
      "logo.jpeg",
      "logo.webp",
    ]
    logo_name = next((name for name in logo_candidates if (destino_saida.parent / name).exists()), None)
    logo_markup = (
        f'<div class="brand-mark"><img src="{html.escape(logo_name)}" alt="Logo da empresa" /></div>'
        if logo_name
        else ""
    )

    html_content = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Simulador de Estratégia Logística</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b1116;
      --panel: #111a21;
      --panel-2: #16232d;
      --text: #e8f1f7;
      --muted: #9ab0c1;
      --ok: #27c36a;
      --bad: #d65050;
      --line: #233543;
      --accent: #48d3a6;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(1200px 500px at 85% -10%, #1b2d39 0%, transparent 60%),
        radial-gradient(900px 450px at -10% -20%, #133129 0%, transparent 55%),
        var(--bg);
      font-family: "Manrope", "Segoe UI", sans-serif;
    }}

    .wrap {{
      max-width: 1450px;
      margin: 0 auto;
      padding: 28px 20px 42px;
    }}

    .top-layout {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      margin-bottom: 16px;
      animation: rise 0.5s ease;
    }}

    .hero-main {{
      display: flex;
      align-items: center;
      gap: 18px;
      min-width: 0;
      padding: 18px 20px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(180deg, #17242e 0%, #111a21 100%);
      box-shadow: 0 18px 38px rgba(0, 0, 0, 0.25);
    }}

    .brand-mark {{
      width: 180px;
      height: 86px;
      border-radius: 20px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      overflow: hidden;
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.18);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 14px 30px rgba(0, 0, 0, 0.22);
    }}

    .brand-mark img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      object-position: center;
      display: block;
    }}

    .hero-copy {{
      min-width: 0;
      flex: 1;
      text-align: center;
    }}

    .hero-copy h1 {{
      margin: 0;
      font-size: clamp(1.4rem, 3vw, 2rem);
      letter-spacing: 0.02em;
    }}

    .hero-copy p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .top-main {{
      display: grid;
      gap: 14px;
    }}

    .control label {{
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    select {{
      width: 100%;
      border: 1px solid #2a3f4f;
      border-radius: 9px;
      padding: 10px;
      background: #0f171d;
      color: var(--text);
      outline: none;
    }}

    .sim-panel {{
      padding: 14px;
      border-radius: 14px;
      border: 1px solid #2a3f4f;
      background: linear-gradient(180deg, #13212a 0%, #0f1a21 100%);
    }}

    .sim-panel h2 {{
      margin: 0 0 6px;
      font-size: 1.05rem;
      color: #c2f6e1;
    }}

    .sim-panel p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.86rem;
    }}

    .sim-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) minmax(140px, 0.7fr);
      gap: 12px;
      align-items: end;
    }}

    .button-control {{
      grid-column: 1 / -1;
      justify-self: center;
      width: min(100%, 340px);
    }}

    .sim-grid input {{
      width: 100%;
      border: 1px solid #2a3f4f;
      border-radius: 9px;
      padding: 10px;
      background: #0f171d;
      color: var(--text);
      outline: none;
    }}

    .sim-btn {{
      width: 100%;
      border: 1px solid #2a6548;
      border-radius: 10px;
      padding: 10px 12px;
      font-weight: 700;
      background: linear-gradient(120deg, #1b4d37, #163d2c);
      color: #c8ffe0;
      cursor: pointer;
    }}

    .sim-btn:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}

    .sim-result {{
      margin-top: 12px;
      border: 1px solid #224d39;
      border-radius: 10px;
      padding: 10px 12px;
      background: rgba(16, 37, 29, 0.7);
      display: none;
    }}

    .sim-result.visible {{
      display: block;
    }}

    .sim-cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 8px;
    }}

    .sim-card {{
      border: 1px solid #2a3f4f;
      border-radius: 12px;
      background: rgba(11, 22, 30, 0.82);
      padding: 10px 12px;
    }}

    .sim-card.top {{
      border-color: rgba(39, 195, 106, 0.55);
      box-shadow: inset 4px 0 0 #27c36a;
    }}

    .sim-card-head {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      margin-bottom: 10px;
      text-align: center;
    }}

    .sim-card-title {{
      font-size: 0.92rem;
      font-weight: 800;
      color: #d7fff0;
    }}

    .sim-card-total {{
      font-size: 1rem;
      font-weight: 800;
      color: #c9ffd8;
    }}

    .sim-card-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0;
      margin-bottom: 8px;
      border: 1px dashed rgba(154, 176, 193, 0.2);
      border-radius: 10px;
      overflow: hidden;
    }}

    .sim-card-meta .sim-kv {{
      flex-direction: column;
      align-items: center;
      gap: 2px;
      border-bottom: none;
      padding: 6px 8px;
      text-align: center;
    }}

    .sim-card-meta .sim-kv + .sim-kv {{
      border-left: 1px dashed rgba(154, 176, 193, 0.2);
    }}

    .sim-card-meta .sim-kv .v {{
      text-align: center;
    }}

    .sim-kv {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 0.82rem;
      border-bottom: 1px dashed rgba(154, 176, 193, 0.2);
      padding-bottom: 3px;
    }}

    .sim-kv .k {{
      color: var(--muted);
    }}

    .sim-kv .v {{
      color: var(--text);
      font-weight: 700;
      text-align: right;
    }}

    .sim-etapas {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }}

    @media (max-width: 1200px) {{
      .sim-cards {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    .sim-etapa {{
      border: 1px solid #223643;
      border-radius: 10px;
      padding: 8px 10px;
      background: rgba(15, 27, 35, 0.75);
    }}

    .sim-etapa h4 {{
      margin: 0 0 6px;
      font-size: 0.82rem;
      color: #b8f8d3;
      letter-spacing: 0.02em;
    }}

    .util-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 82px;
      padding: 4px 8px;
      border-radius: 999px;
      font-weight: 800;
      font-size: 0.78rem;
      border: 1px solid transparent;
    }}

    .util-good {{
      background: rgba(39, 195, 106, 0.16);
      color: #bff5d2;
      border-color: rgba(39, 195, 106, 0.45);
    }}

    .util-warn {{
      background: rgba(245, 194, 66, 0.16);
      color: #ffe8a3;
      border-color: rgba(245, 194, 66, 0.45);
    }}

    .util-bad {{
      background: rgba(214, 80, 80, 0.16);
      color: #ffc4c4;
      border-color: rgba(214, 80, 80, 0.45);
    }}

    @keyframes rise {{
      from {{ transform: translateY(8px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}

    @media (max-width: 800px) {{
      .wrap {{ padding: 18px 10px 28px; }}
      .hero-main {{ align-items: center; flex-direction: column; }}
      .hero-copy {{ width: 100%; text-align: center; }}
      .brand-mark {{ width: 150px; height: 72px; border-radius: 16px; }}
      .brand-mark img {{ width: 100%; height: 100%; }}
      .sim-grid {{ grid-template-columns: 1fr; }}
      .button-control {{ grid-column: auto; justify-self: stretch; width: 100%; }}
      .sim-cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="top-layout">
      <div class="top-main">
        <div class="hero-main">
          {logo_markup}
          <div class="hero-copy">
            <h1>Simulador de Estratégia Logística</h1>
            <p>Compare operações DIRETO e HUB com foco em custo, capacidade e ocupação.</p>
          </div>
        </div>

        <section class="sim-panel">
          <h2>Planejamento da Carga</h2>
          <p>{html.escape(simulador_msg)}</p>
          <div class="sim-grid">
            <div class="control">
              <label for="simProduto">Produto</label>
              <select id="simProduto" {'disabled' if not simulador_ativo else ''}>
                {produtos_options if produtos_options else '<option value="">Sem produtos disponíveis</option>'}
              </select>
            </div>
            <div class="control">
              <label for="simDestino">Destino</label>
              <select id="simDestino" {'disabled' if not simulador_ativo else ''}>
                {_options_html(destinos, include_all=False)}
              </select>
            </div>
            <div class="control">
              <label for="simQuantidade">Quantidade</label>
              <input id="simQuantidade" type="number" min="1" step="1" value="100" {'disabled' if not simulador_ativo else ''} />
            </div>
            <div class="control button-control">
              <button class="sim-btn" id="simularBtn" {'disabled' if not simulador_ativo else ''}>Simular Melhor Estratégia</button>
            </div>
          </div>
        </section>
      </div>

    </section>

    <section class="sim-panel">
      <div class="sim-result" id="simResult"></div>
    </section>
  </div>

  <script>
    const produtosData = {json.dumps(produtos_data, ensure_ascii=False)};
    const capacidadesData = {json.dumps(capacidades_data, ensure_ascii=False)};
    const rotasData = {json.dumps(rotas_data, ensure_ascii=False)};

    function formatBRL(value) {{
      return new Intl.NumberFormat('pt-BR', {{ style: 'currency', currency: 'BRL' }}).format(Number(value) || 0);
    }}

    function formatKG(value) {{
      return `${{Number(value || 0).toLocaleString('pt-BR')}} kg`;
    }}

    function formatPercent(value) {{
      return `${{Number(value || 0).toLocaleString('pt-BR', {{ minimumFractionDigits: 1, maximumFractionDigits: 1 }})}}%`;
    }}

    function getUtilClass(value) {{
      if (Number(value) > 75) return 'util-good';
      if (Number(value) >= 50) return 'util-warn';
      return 'util-bad';
    }}

    function estimatePalletCapacity(capacidadePeso, palletCapOriginal, palletsTotais, pesoBruto) {{
      if (Number(palletCapOriginal) > 0) return Number(palletCapOriginal);
      if (!(capacidadePeso > 0) || !(palletsTotais > 0) || !(pesoBruto > 0)) return 0;
      const mediaPesoPorPallet = pesoBruto / palletsTotais;
      if (!(mediaPesoPorPallet > 0)) return 0;
      return Math.max(1, Math.floor(capacidadePeso / mediaPesoPorPallet));
    }}

    function montarEtapa(transportadora, nVeiculos, capacidadePesoTotal, capacidadePalletTotal, pesoUtilizado, palletsUtilizados, custo) {{
      if (!transportadora || transportadora === '-') return null;
      const ocupPeso = capacidadePesoTotal > 0 ? (pesoUtilizado / capacidadePesoTotal) * 100 : 0;
      const ocupPallet = capacidadePalletTotal > 0 ? (palletsUtilizados / capacidadePalletTotal) * 100 : 0;
      const gargalo = capacidadePalletTotal > 0 && ocupPallet >= ocupPeso ? 'PALLET' : 'PESO';
      return {{
        transportadora,
        nVeiculos,
        pesoUtilizado,
        capacidadePesoTotal,
        palletsUtilizados,
        capacidadePalletTotal,
        ocupPeso,
        ocupPallet,
        gargalo,
        custo,
      }};
    }}

    function getEtapaUtilizacaoLimitante(etapa) {{
      if (!etapa) return 100;
      return Math.max(Number(etapa.ocupPeso || 0), Number(etapa.ocupPallet || 0));
    }}

    function calcularScoreCenario(cenario) {{
      const etapas = [cenario.etapa1, cenario.etapa2].filter(Boolean);
      const utilizacoes = etapas.map(getEtapaUtilizacaoLimitante);
      return utilizacoes.length ? Math.min(...utilizacoes) : 0;
    }}

    function filtrarCenariosCompetitivos(cenarios) {{
      const dedupe = new Map();
      cenarios.forEach(cenario => {{
        cenario.utilScore = calcularScoreCenario(cenario);
        const key = [
          cenario.estrategia,
          cenario.hub,
          cenario.formacao,
          cenario.etapa1?.transportadora || '-',
          cenario.etapa2?.transportadora || '-',
        ].join('|');
        const atual = dedupe.get(key);
        if (!atual || cenario.custoTotal < atual.custoTotal) {{
          dedupe.set(key, cenario);
        }}
      }});

      const base = Array.from(dedupe.values()).sort(
        (a, b) => (a.custoTotal - b.custoTotal) || (a.qtdVeiculos - b.qtdVeiculos) || (b.utilScore - a.utilScore)
      );

      const naoDominados = base.filter((cenario, idx) => !base.some((outro, otherIdx) => {{
        if (idx === otherIdx) return false;
        const dominaEmCusto = Number(outro.custoTotal) <= Number(cenario.custoTotal);
        const dominaEmVeiculos = Number(outro.qtdVeiculos) <= Number(cenario.qtdVeiculos);
        const dominaEmUtil = Number(outro.utilScore) >= Number(cenario.utilScore);
        const estritamenteMelhor =
          Number(outro.custoTotal) < Number(cenario.custoTotal) ||
          Number(outro.qtdVeiculos) < Number(cenario.qtdVeiculos) ||
          Number(outro.utilScore) > Number(cenario.utilScore);
        return dominaEmCusto && dominaEmVeiculos && dominaEmUtil && estritamenteMelhor;
      }}));

      if (!naoDominados.length) return base.slice(0, 5);

      const melhorCusto = Math.min(...naoDominados.map(c => Number(c.custoTotal)));
      const menorQtdVeiculos = Math.min(...naoDominados.map(c => Number(c.qtdVeiculos)));
      const tetoVeiculos = Math.max(menorQtdVeiculos + 2, Math.ceil(menorQtdVeiculos * 1.4));
      const tetoCusto = melhorCusto * 1.18;

      const competitivos = naoDominados.filter(cenario => {{
        const custoCompetitivo = Number(cenario.custoTotal) <= tetoCusto;
        const veiculosCompetitivo = Number(cenario.qtdVeiculos) <= tetoVeiculos;
        const utilizacaoAceitavel = Number(cenario.utilScore) >= 50;
        return (custoCompetitivo || veiculosCompetitivo) && utilizacaoAceitavel;
      }});

      const ordenadosCompetitivos = (competitivos.length ? competitivos : naoDominados.slice(0, 5)).sort(
        (a, b) => (a.custoTotal - b.custoTotal) || (a.qtdVeiculos - b.qtdVeiculos) || (b.utilScore - a.utilScore)
      );

      const escolhidos = [];
      const vistos = new Set();
      const tentarAdicionar = (lista) => {{
        lista.forEach(cenario => {{
          if (escolhidos.length >= 3) return;
          const chave = [
            cenario.estrategia,
            cenario.hub,
            cenario.formacao,
            cenario.etapa1?.transportadora || '-',
            cenario.etapa2?.transportadora || '-',
          ].join('|');
          if (vistos.has(chave)) return;
          vistos.add(chave);
          escolhidos.push(cenario);
        }});
      }};

      tentarAdicionar(ordenadosCompetitivos);
      if (escolhidos.length < 3) {{
        const ordenadosNaoDominados = naoDominados.slice().sort(
          (a, b) => (a.custoTotal - b.custoTotal) || (a.qtdVeiculos - b.qtdVeiculos) || (b.utilScore - a.utilScore)
        );
        tentarAdicionar(ordenadosNaoDominados);
      }}
      if (escolhidos.length < 3) {{
        tentarAdicionar(base);
      }}

      return escolhidos.slice(0, 3);
    }}

    function mapearCapacidade(veiculoId, transportadora) {{
      const transp = String(transportadora || '').trim().toUpperCase();
      const veic = String(veiculoId || '').trim().toUpperCase();

      const byTransportadora = capacidadesData.filter(c => String(c.operador || '').toUpperCase().includes(transp));

      const veicNum = Number(veic);
      if (!Number.isNaN(veicNum) && veic !== '') {{
        const candidatos = (byTransportadora.length ? byTransportadora : capacidadesData)
          .filter(c => Number(c.peso_max) >= veicNum * 0.85 && Number(c.peso_max) <= veicNum * 1.15);
        if (candidatos.length) return candidatos[0];
      }}

      const candidatosTexto = (byTransportadora.length ? byTransportadora : capacidadesData)
        .filter(c => String(c.tipo || '').toUpperCase().includes(veic));
      if (candidatosTexto.length) return candidatosTexto[0];

      if (!Number.isNaN(veicNum) && veic !== '') {{
        return {{ tipo: veic, peso_max: veicNum, paletes: null, operador: transp }};
      }}

      return null;
    }}

    function simularCargaFrontend() {{
      const resultEl = document.getElementById('simResult');
      const produtoNome = document.getElementById('simProduto').value;
      const destino = String(document.getElementById('simDestino').value || '').trim().toUpperCase();
      const quantidade = Number(document.getElementById('simQuantidade').value);

      if (!produtoNome || !destino || !(quantidade > 0)) {{
        resultEl.classList.add('visible');
        resultEl.innerHTML = '<strong>Preencha produto, destino e quantidade.</strong>';
        return;
      }}

      const produto = produtosData.find(p => String(p.produto || '').toUpperCase() === produtoNome.toUpperCase());
      if (!produto) {{
        resultEl.classList.add('visible');
        resultEl.innerHTML = '<strong>Produto não encontrado na base.</strong>';
        return;
      }}

      const pesoLiquido = quantidade * Number(produto.peso_unitario || 0);
      const fatorBruto = Number(produto.fator_peso_bruto || 1) || 1;
      const pesoTotal = pesoLiquido * fatorBruto;
      const qtdPorPallet = Number(produto.qtd_por_pallet || 1);
      const palletsTotais = Math.ceil(quantidade / Math.max(qtdPorPallet, 1));

      const rotasDestino = rotasData.filter(r => String(r.destino || '') === destino);
      if (!rotasDestino.length) {{
        resultEl.classList.add('visible');
        resultEl.innerHTML = '<strong>Não há rotas para o destino informado.</strong>';
        return;
      }}

      const melhoresPorVeiculo = [];
      const seen = new Set();
      rotasDestino
        .slice()
        .sort((a, b) => Number(a.total) - Number(b.total))
        .forEach(r => {{
          const key = String(r.veiculo1 || '').toUpperCase();
          if (!seen.has(key)) {{
            seen.add(key);
            melhoresPorVeiculo.push(r);
          }}
        }});

      const cenarios = [];
      melhoresPorVeiculo.forEach(r => {{
        const cap1 = mapearCapacidade(r.veiculo1, r.transp1);
        const cap2 = r.transp2 ? (mapearCapacidade(r.veiculo2 || r.veiculo1, r.transp2) || cap1) : null;
        if (!cap1 || !(Number(cap1.peso_max) > 0)) return;

        const palletCap1 = estimatePalletCapacity(Number(cap1.peso_max), Number(cap1.paletes || 0), palletsTotais, pesoTotal);
        const palletCap2 = cap2 ? estimatePalletCapacity(Number(cap2.peso_max), Number(cap2.paletes || 0), palletsTotais, pesoTotal) : 0;

        let n = Math.ceil(pesoTotal / Number(cap1.peso_max));
        if (palletCap1 > 0) {{
          n = Math.max(n, Math.ceil(palletsTotais / palletCap1));
        }}
        if (cap2 && Number(cap2.peso_max) > 0) {{
          n = Math.max(n, Math.ceil(pesoTotal / Number(cap2.peso_max)));
          if (palletCap2 > 0) {{
            n = Math.max(n, Math.ceil(palletsTotais / palletCap2));
          }}
        }}

        if ((pesoTotal / Math.max(n, 1)) > Number(cap1.peso_max)) return;
        if (cap2 && (pesoTotal / Math.max(n, 1)) > Number(cap2.peso_max)) return;

        cenarios.push({{
          origem: r.origem || 'PORTO STS',
          destino,
          estrategia: r.rota,
          hub: r.hub || '-',
          formacao: `${{n}}x ${{Number(cap1.peso_max).toLocaleString('pt-BR')}} kg`,
          etapa1: montarEtapa(
            r.transp1 || '-',
            n,
            n * Number(cap1.peso_max),
            n * Number(palletCap1),
            pesoTotal,
            palletsTotais,
            n * Number(r.custo1 || 0),
          ),
          etapa2: cap2 && (r.transp2 || Number(r.custo2 || 0) > 0)
            ? montarEtapa(
                r.transp2 || '-',
                n,
                n * Number(cap2.peso_max),
                n * Number(palletCap2),
                pesoTotal,
                palletsTotais,
                n * Number(r.custo2 || 0),
              )
            : null,
          qtdVeiculos: n,
          custoTotal: Number(r.total) * n,
        }});
      }});

      for (let i = 0; i < melhoresPorVeiculo.length; i += 1) {{
        for (let j = i + 1; j < melhoresPorVeiculo.length; j += 1) {{
          const r1 = melhoresPorVeiculo[i];
          const r2 = melhoresPorVeiculo[j];
          if (r1.rota !== r2.rota) continue;
          if (r1.rota === 'HUB' && r1.hub !== r2.hub) continue;
          const c1 = mapearCapacidade(r1.veiculo1, r1.transp1);
          const c2 = mapearCapacidade(r2.veiculo1, r2.transp1);
          if (!c1 || !c2) continue;
          if (!(Number(c1.peso_max) > 0) || !(Number(c2.peso_max) > 0)) continue;

          const c1Etapa2 = r1.transp2 ? (mapearCapacidade(r1.veiculo2 || r1.veiculo1, r1.transp2) || c1) : null;
          const c2Etapa2 = r2.transp2 ? (mapearCapacidade(r2.veiculo2 || r2.veiculo1, r2.transp2) || c2) : null;
          const p1 = estimatePalletCapacity(Number(c1.peso_max), Number(c1.paletes || 0), palletsTotais, pesoTotal);
          const p2 = estimatePalletCapacity(Number(c2.peso_max), Number(c2.paletes || 0), palletsTotais, pesoTotal);
          const p1Etapa2 = c1Etapa2 ? estimatePalletCapacity(Number(c1Etapa2.peso_max), Number(c1Etapa2.paletes || 0), palletsTotais, pesoTotal) : 0;
          const p2Etapa2 = c2Etapa2 ? estimatePalletCapacity(Number(c2Etapa2.peso_max), Number(c2Etapa2.paletes || 0), palletsTotais, pesoTotal) : 0;

          for (let n1 = 1; n1 <= 5; n1 += 1) {{
            for (let n2 = 1; n2 <= 5; n2 += 1) {{
              if ((n1 + n2) > 7) continue;

              const pesoCap = (n1 * Number(c1.peso_max)) + (n2 * Number(c2.peso_max));
              if (pesoCap < pesoTotal) continue;
              if ((p1 > 0 || p2 > 0) && ((n1 * p1) + (n2 * p2) < palletsTotais)) continue;

              const pesoCapEtapa2 = (n1 * Number(c1Etapa2?.peso_max || 0)) + (n2 * Number(c2Etapa2?.peso_max || 0));
              const palletCapEtapa2 = (n1 * Number(p1Etapa2 || 0)) + (n2 * Number(p2Etapa2 || 0));
              if ((c1Etapa2 || c2Etapa2) && pesoCapEtapa2 < pesoTotal) continue;
              if ((p1Etapa2 > 0 || p2Etapa2 > 0) && palletCapEtapa2 < palletsTotais) continue;

              cenarios.push({{
                origem: r1.origem || 'PORTO STS',
                destino,
                estrategia: r1.rota,
                hub: r1.hub || '-',
                formacao: `${{n1}}x ${{Number(c1.peso_max).toLocaleString('pt-BR')}} kg + ${{n2}}x ${{Number(c2.peso_max).toLocaleString('pt-BR')}} kg`,
                etapa1: montarEtapa(
                  `${{r1.transp1 || '-'}} / ${{r2.transp1 || '-'}}`,
                  n1 + n2,
                  (n1 * Number(c1.peso_max)) + (n2 * Number(c2.peso_max)),
                  (n1 * Number(p1 || 0)) + (n2 * Number(p2 || 0)),
                  pesoTotal,
                  palletsTotais,
                  (n1 * Number(r1.custo1 || 0)) + (n2 * Number(r2.custo1 || 0)),
                ),
                etapa2: (c1Etapa2 || c2Etapa2)
                  ? montarEtapa(
                      `${{r1.transp2 || '-'}} / ${{r2.transp2 || '-'}}`,
                      n1 + n2,
                      pesoCapEtapa2,
                      palletCapEtapa2,
                      pesoTotal,
                      palletsTotais,
                      (n1 * Number(r1.custo2 || 0)) + (n2 * Number(r2.custo2 || 0)),
                    )
                  : null,
                qtdVeiculos: n1 + n2,
                custoTotal: (n1 * Number(r1.total)) + (n2 * Number(r2.total)),
              }});
            }}
          }}

          // Spillover: N veículos principais + 1 menor p/ excedente (cobre volumes acima de 5 caminhões)
          [
            [Math.ceil(pesoTotal / Number(c1.peso_max)) - 1, 1],
            [Math.ceil(pesoTotal / Number(c1.peso_max)), 1],
            [1, Math.ceil(pesoTotal / Number(c2.peso_max)) - 1],
            [1, Math.ceil(pesoTotal / Number(c2.peso_max))],
          ].forEach(([n1, n2]) => {{
            if (n1 < 1 || n2 < 1) return;
            if (n1 <= 5 && n2 <= 5 && (n1 + n2) <= 7) return; // já coberto pelos loops acima

            const pesoCap = (n1 * Number(c1.peso_max)) + (n2 * Number(c2.peso_max));
            if (pesoCap < pesoTotal) return;
            if ((p1 > 0 || p2 > 0) && ((n1 * p1) + (n2 * p2) < palletsTotais)) return;

            const pesoCapEtapa2 = (n1 * Number(c1Etapa2?.peso_max || 0)) + (n2 * Number(c2Etapa2?.peso_max || 0));
            const palletCapEtapa2 = (n1 * Number(p1Etapa2 || 0)) + (n2 * Number(p2Etapa2 || 0));
            if ((c1Etapa2 || c2Etapa2) && pesoCapEtapa2 < pesoTotal) return;
            if ((p1Etapa2 > 0 || p2Etapa2 > 0) && palletCapEtapa2 < palletsTotais) return;

            cenarios.push({{
              origem: r1.origem || 'PORTO STS',
              destino,
              estrategia: r1.rota,
              hub: r1.hub || '-',
              formacao: `${{n1}}x ${{Number(c1.peso_max).toLocaleString('pt-BR')}} kg + ${{n2}}x ${{Number(c2.peso_max).toLocaleString('pt-BR')}} kg`,
              etapa1: montarEtapa(
                `${{r1.transp1 || '-'}} / ${{r2.transp1 || '-'}}`,
                n1 + n2,
                (n1 * Number(c1.peso_max)) + (n2 * Number(c2.peso_max)),
                (n1 * Number(p1 || 0)) + (n2 * Number(p2 || 0)),
                pesoTotal,
                palletsTotais,
                (n1 * Number(r1.custo1 || 0)) + (n2 * Number(r2.custo1 || 0)),
              ),
              etapa2: (c1Etapa2 || c2Etapa2)
                ? montarEtapa(
                    `${{r1.transp2 || '-'}} / ${{r2.transp2 || '-'}}`,
                    n1 + n2,
                    pesoCapEtapa2,
                    palletCapEtapa2,
                    pesoTotal,
                    palletsTotais,
                    (n1 * Number(r1.custo2 || 0)) + (n2 * Number(r2.custo2 || 0)),
                  )
                : null,
              qtdVeiculos: n1 + n2,
              custoTotal: (n1 * Number(r1.total)) + (n2 * Number(r2.total)),
            }});
          }});
        }}
      }}

      if (!cenarios.length) {{
        resultEl.classList.add('visible');
        resultEl.innerHTML = '<strong>Nenhum cenário viável com os limites de peso/pallets.</strong>';
        return;
      }}

      const cenariosCompetitivos = filtrarCenariosCompetitivos(cenarios);
      cenariosCompetitivos.sort((a, b) => (a.custoTotal - b.custoTotal) || (a.qtdVeiculos - b.qtdVeiculos) || (b.utilScore - a.utilScore));
      resultEl.classList.add('visible');
      const renderEtapaCard = (titulo, etapa) => {{
        if (!etapa) return '';
        return `
          <section class="sim-etapa">
            <h4>${{titulo}}</h4>
            <div class="sim-kv"><span class="k">Transportadora</span><span class="v">${{etapa.transportadora}}</span></div>
            <div class="sim-kv"><span class="k">Veículos</span><span class="v">${{Number(etapa.nVeiculos || 0).toLocaleString('pt-BR')}}</span></div>
            <div class="sim-kv"><span class="k">Peso</span><span class="v">${{Number(etapa.pesoUtilizado || 0).toLocaleString('pt-BR')}} / ${{Number(etapa.capacidadePesoTotal || 0).toLocaleString('pt-BR')}} kg</span></div>
            <div class="sim-kv"><span class="k">Pallet</span><span class="v">${{Number(etapa.palletsUtilizados || 0).toLocaleString('pt-BR')}} / ${{Number(etapa.capacidadePalletTotal || 0).toLocaleString('pt-BR')}}</span></div>
            <div class="sim-kv"><span class="k">Ocupação Peso</span><span class="v"><span class="util-pill ${{getUtilClass(etapa.ocupPeso)}}">${{formatPercent(etapa.ocupPeso)}}</span></span></div>
            <div class="sim-kv"><span class="k">Ocupação Pallet</span><span class="v"><span class="util-pill ${{getUtilClass(etapa.ocupPallet)}}">${{formatPercent(etapa.ocupPallet)}}</span></span></div>
            <div class="sim-kv"><span class="k">Custo</span><span class="v">${{formatBRL(etapa.custo)}}</span></div>
          </section>
        `;
      }};

      const montarTituloCenario = (cenario, idx) => {{
        const origemRota = String(cenario.origem || 'PORTO STS').trim().toUpperCase();
        const hubRota = String(cenario.hub || '').trim().toUpperCase();
        const destinoRota = String(cenario.destino || destino || '-').trim().toUpperCase();
        const pontos = [origemRota];
        if (hubRota && hubRota !== '-') pontos.push(hubRota);
        pontos.push(destinoRota);
        return `#${{idx + 1}} • ${{pontos.join(' -> ')}}`;
      }};

      const cards = cenariosCompetitivos.map((c, idx) => `
        <article class="sim-card ${{idx === 0 ? 'top' : ''}}">
          <div class="sim-card-head">
            <div class="sim-card-title">${{montarTituloCenario(c, idx)}}</div>
            <div class="sim-card-total">${{formatBRL(c.custoTotal)}}</div>
          </div>
          <div class="sim-card-meta">
            <div class="sim-kv"><span class="k">Formação</span><span class="v">${{c.formacao}}</span></div>
            <div class="sim-kv"><span class="k">Score de utilização</span><span class="v">${{formatPercent(c.utilScore || 0)}}</span></div>
          </div>
          <div class="sim-etapas">
            ${{renderEtapaCard('ETAPA 1', c.etapa1)}}
            ${{renderEtapaCard('ETAPA 2', c.etapa2)}}
          </div>
        </article>
      `).join('');

      resultEl.innerHTML = `
        <div class="sim-kv" style="margin-bottom:8px; border-bottom:none;">
          <span class="k">Exibindo opções competitivas</span>
          <span class="v">Top ${{cenariosCompetitivos.length}}</span>
        </div>
        <div class="sim-cards">${{cards}}</div>
      `;
    }}

    const simularBtn = document.getElementById('simularBtn');
    const simProduto = document.getElementById('simProduto');
    const simDestino = document.getElementById('simDestino');
    const simQuantidade = document.getElementById('simQuantidade');
    if (simularBtn) {{
      simularBtn.addEventListener('click', simularCargaFrontend);
    }}


  </script>
</body>
</html>
"""

    destino_saida.write_text(html_content, encoding="utf-8")


# ================================================================
# SIMULADOR DE CARGA — Camada de simulação sobre o motor principal
# ================================================================


def carregar_parametros_produtos(path: Path) -> pd.DataFrame:
    """Lê a base de produtos (Parametros_Produtos.xlsx) e normaliza colunas."""
    df = pd.read_excel(path)
    df.columns = [normalizar_nome_coluna(c) for c in df.columns]
    return df


def carregar_capacidade_veiculos(path: Path) -> pd.DataFrame:
    """Lê a tabela de capacidade por veículo/transportadora e normaliza colunas."""
    df = pd.read_excel(path)
    df.columns = [normalizar_nome_coluna(c) for c in df.columns]
    df["OPERADOR"] = df["OPERADOR"].astype(str).str.strip().str.upper()
    df["TIPO VEICULO"] = df["TIPO VEICULO"].astype(str).str.strip().str.upper()
    return df


def _encontrar_produto(nome_ou_codigo: str, df_produtos: pd.DataFrame) -> "pd.Series | None":
    """Localiza um produto na base por código ou nome (exato ou parcial)."""
    chave = nome_ou_codigo.strip().upper()
    for col in ("CODIGO", "PRODUTO"):
        if col not in df_produtos.columns:
            continue
        mask = df_produtos[col].astype(str).str.strip().str.upper() == chave
        if mask.any():
            return df_produtos[mask].iloc[0]
    # Busca parcial por nome
    if "PRODUTO" in df_produtos.columns:
        mask = df_produtos["PRODUTO"].astype(str).str.upper().str.contains(chave, regex=False, na=False)
        if mask.any():
            return df_produtos[mask].iloc[0]
    return None


def _capacidade_veiculo(
    veiculo_id: str,
    transp: str,
    df_cap: "pd.DataFrame | None",
) -> "dict | None":
    """Retorna dict com 'tipo', 'peso_max' e 'paletes' para o veículo dado.

    Estratégias de matching (em ordem de prioridade):
    1. VEICULO_ID numérico → match por Peso MAXIMO ± 15%, filtrado por transportadora.
    2. VEICULO_ID texto    → match por Tipo Veiculo contém, filtrado por transportadora.
    3. Fallback            → sem df_cap, tenta interpretar o id como peso em kg.
    """
    if df_cap is None or df_cap.empty:
        try:
            peso = float(veiculo_id)
            return {"tipo": veiculo_id, "peso_max": peso, "paletes": None}
        except (ValueError, TypeError):
            return None

    transp_up = str(transp).strip().upper()
    veic_up = str(veiculo_id).strip().upper()

    def _to_row(sub: pd.DataFrame) -> "dict | None":
        if sub.empty:
            return None
        row = sub.iloc[0]
        paletes_val = row["PALETES"] if "PALETES" in row.index else None
        return {
            "tipo": str(row["TIPO VEICULO"]),
            "peso_max": float(row["PESO MAXIMO POR VEICULO"]),
            "paletes": int(paletes_val) if pd.notna(paletes_val) else None,
        }

    # 1. Numérico
    try:
        peso_num = float(veiculo_id)
        col_peso = pd.to_numeric(df_cap["PESO MAXIMO POR VEICULO"], errors="coerce")
        mask_peso = col_peso.between(peso_num * 0.85, peso_num * 1.15)
        sub = df_cap[mask_peso]
        sub_transp = sub[sub["OPERADOR"].str.contains(transp_up, regex=False, na=False)]
        result = _to_row(sub_transp) or _to_row(sub)
        if result:
            return result
    except (ValueError, TypeError):
        pass

    # 2. Texto
    mask_tipo = df_cap["TIPO VEICULO"].str.contains(veic_up, regex=False, na=False)
    sub = df_cap[mask_tipo]
    sub_transp = sub[sub["OPERADOR"].str.contains(transp_up, regex=False, na=False)]
    result = _to_row(sub_transp) or _to_row(sub)
    if result:
        return result

    return None


@dataclass
class ResultadoSimulacao:
    produto: str
    destino: str
    quantidade: float
    peso_total: float
    pallets_totais: int
    estrategia: str
    formacao_veiculos: str
    transportadora: str
    custo_total: float
    quantidade_veiculos: int
    custo_por_veiculo: float
    cenarios_avaliados: int
    insight: str


def simular_carga(
    produto: str,
    destino: str,
    quantidade: float,
    df_produtos: pd.DataFrame,
    df_inbound: pd.DataFrame,
    df_transferencia: pd.DataFrame,
    df_capacidade: "pd.DataFrame | None" = None,
    hubs: "List[str] | None" = None,
) -> ResultadoSimulacao:
    """Simula a melhor estratégia de transporte para produto + destino + quantidade.

    Etapas:
    1. Localiza produto e calcula peso_total e pallets.
    2. Executa o motor atual (gerar_tabela_consolidada) e filtra pelo destino.
    3. Para cada tipo de veículo disponível, calcula quantas unidades são necessárias.
    4. Gera cenários homogêneos e mistos (até 2 tipos de veículo combinados).
    5. Descarta cenários que violam limites de peso ou pallets.
    6. Retorna o cenário de menor custo total com insight textual.
    """
    if hubs is None:
        hubs = HUBS_PADRAO

    destino_norm = destino.strip().upper()

    # ── 1. Produto ─────────────────────────────────────────────────────────
    produto_row = _encontrar_produto(produto, df_produtos)
    if produto_row is None:
        raise ValueError(f"Produto '{produto}' não encontrado na base de produtos.")

    nome_produto = str(produto_row.get("PRODUTO", produto))
    peso_unit = float(produto_row.get("PESO LIQUIDO", 1.0))
    qtd_pallet = float(produto_row.get("QUANTIDADE POR PALLET", 1.0))

    peso_total = quantidade * peso_unit
    pallets_totais = _math_mod.ceil(quantidade / qtd_pallet)

    # ── 2. Motor atual ─────────────────────────────────────────────────────
    cols = ColunasPadrao()
    consolidado = gerar_tabela_consolidada(df_inbound, df_transferencia, hubs, cols)

    rotas_destino = consolidado[
        consolidado["DESTINO"].astype(str).str.strip().str.upper() == destino_norm
    ].copy()

    if rotas_destino.empty:
        raise ValueError(f"Destino '{destino}' não encontrado nas tabelas de frete.")

    # Melhor rota por tipo de veículo (menor custo total)
    melhores_por_veiculo = (
        rotas_destino
        .sort_values("TOTAL")
        .drop_duplicates(subset=["VEICULO 1"], keep="first")
        .reset_index(drop=True)
    )

    # ── 3 & 4. Gerar cenários ─────────────────────────────────────────────
    cenarios: List[dict] = []

    veiculos_info = []
    for _, rota in melhores_por_veiculo.iterrows():
        veic_id = str(rota["VEICULO 1"])
        transp = str(rota["TRANSP. 1"])
        custo_unit = float(rota["TOTAL"])
        cap = _capacidade_veiculo(veic_id, transp, df_capacidade)
        veiculos_info.append((rota, cap, custo_unit))

    # Cenários homogêneos (apenas um tipo de veículo)
    for rota, cap, custo_unit in veiculos_info:
        if cap is None:
            continue
        peso_max = cap["peso_max"]
        paletes_max = cap["paletes"]

        n = _math_mod.ceil(peso_total / peso_max)
        if paletes_max:
            n = max(n, _math_mod.ceil(pallets_totais / paletes_max))

        # Cada veículo não pode ser sobrecarregado
        if (peso_total / n) > peso_max:
            continue

        cenarios.append({
            "tipo": cap["tipo"],
            "n_veiculos": n,
            "formacao": f"{n}x {cap['tipo']}",
            "transp": str(rota["TRANSP. 1"]),
            "custo_unit": custo_unit,
            "custo_total": custo_unit * n,
            "rota_tipo": str(rota["ROTA"]),
            "hub": str(rota.get("HUB", "")),
            "misto": False,
        })

    # Cenários mistos (combinação de 2 tipos de veículo)
    for i in range(len(veiculos_info)):
        for j in range(i + 1, len(veiculos_info)):
            r1, cap1, cu1 = veiculos_info[i]
            r2, cap2, cu2 = veiculos_info[j]
            if cap1 is None or cap2 is None:
                continue
            for n1 in range(1, 6):
                for n2 in range(1, 6):
                    if n1 + n2 > 7:
                        continue
                    peso_ok = (n1 * cap1["peso_max"] + n2 * cap2["peso_max"]) >= peso_total
                    if not peso_ok:
                        continue
                    pal1 = cap1["paletes"] or 0
                    pal2 = cap2["paletes"] or 0
                    pal_ok = True
                    if pal1 > 0 or pal2 > 0:
                        pal_ok = (n1 * pal1 + n2 * pal2) >= pallets_totais
                    if not pal_ok:
                        continue
                    cenarios.append({
                        "tipo": f"{cap1['tipo']}+{cap2['tipo']}",
                        "n_veiculos": n1 + n2,
                        "formacao": f"{n1}x {cap1['tipo']} + {n2}x {cap2['tipo']}",
                        "transp": f"{r1['TRANSP. 1']} / {r2['TRANSP. 1']}",
                        "custo_unit": (cu1 + cu2) / 2,
                        "custo_total": n1 * cu1 + n2 * cu2,
                        "rota_tipo": str(r1["ROTA"]),
                        "hub": str(r1.get("HUB", "")),
                        "misto": True,
                    })

    # ── 5. Descartar inválidos já feito nas validações acima ──────────────

    if not cenarios:
        raise ValueError(
            f"Nenhum cenário logístico viável para '{destino}' "
            f"com {peso_total:,.0f} kg e {pallets_totais} pallet(s). "
            "Verifique os dados de capacidade e as rotas disponíveis."
        )

    # ── 6. Selecionar o melhor cenário ────────────────────────────────────
    cenarios_validos = sorted(cenarios, key=lambda c: (round(c["custo_total"], 2), c["n_veiculos"]))
    melhor = cenarios_validos[0]

    estrategia = "HÍBRIDO" if melhor["misto"] else melhor["rota_tipo"]

    hub_texto = ""
    hub_val = melhor["hub"]
    if hub_val and hub_val not in {"", "nan", "None"}:
        hub_texto = f" via HUB {hub_val}"

    insight = (
        f"Para o destino {destino_norm}, recomendamos {melhor['formacao']}"
        f" ({melhor['transp']}){hub_texto}, pois essa combinação atende "
        f"{pallets_totais} pallet(s) e {peso_total:,.0f} kg "
        f"com o menor custo total de {_formatar_brl(melhor['custo_total'])}."
    )

    return ResultadoSimulacao(
        produto=nome_produto,
        destino=destino_norm,
        quantidade=quantidade,
        peso_total=peso_total,
        pallets_totais=pallets_totais,
        estrategia=estrategia,
        formacao_veiculos=melhor["formacao"],
        transportadora=melhor["transp"],
        custo_total=melhor["custo_total"],
        quantidade_veiculos=melhor["n_veiculos"],
        custo_por_veiculo=melhor["custo_unit"],
        cenarios_avaliados=len(cenarios),
        insight=insight,
    )


def executar(
    inbound_path: Path,
    transferencia_path: Path,
    html_saida: Path,
    csv_saida: Path | None,
    hubs: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ColunasPadrao()

    required_inbound = [
        cols.operacao,
        cols.destino,
        cols.uf_destino,
        cols.tipo_transp,
        cols.transportadora,
        cols.veiculo_kg,
        cols.frete,
    ]
    required_transf = [
        cols.operacao,
        cols.origem,
        cols.destino,
        cols.uf_destino,
        cols.tipo_transp,
        cols.transportadora,
        cols.veiculo_kg,
        cols.frete,
    ]

    inbound = ler_tabela(inbound_path, obrigatorias=required_inbound)
    transferencia = ler_tabela(transferencia_path, obrigatorias=required_transf)

    consolidado = gerar_tabela_consolidada(inbound, transferencia, hubs, cols)

    melhores = (
        consolidado[consolidado["MELHOR_POR_DESTINO_VEICULO"]]
        .sort_values(["DESTINO", "VEICULO 1", "TOTAL", "ROTA", "HUB"])
        .drop_duplicates(subset=["DESTINO", "VEICULO 1"], keep="first")
        .reset_index(drop=True)
    )

    gerar_html(consolidado, html_saida)
    if csv_saida is not None:
        consolidado.to_csv(csv_saida, index=False, sep=";", decimal=",")

    return consolidado, melhores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera dashboard HTML com combinações de frete DIRETO e HUB e menor custo por destino e veículo."
    )
    parser.add_argument(
        "--inbound",
        type=Path,
        default=Path("Frete_Inbound.xlsx"),
        help="Arquivo de frete inbound (.xlsx/.xls/.csv).",
    )
    parser.add_argument(
        "--transferencia",
        type=Path,
        default=Path("Frete_Transferencia.xlsx"),
        help="Arquivo de frete de transferência (.xlsx/.xls/.csv).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("dashboard_rotas_logisticas.html"),
        help="Caminho do HTML final.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("rotas_consolidadas.csv"),
        help="CSV consolidado (opcional). Use vazio para não gerar.",
    )
    parser.add_argument(
        "--hubs",
        nargs="+",
        default=HUBS_PADRAO,
        help="Lista de CDs para rota HUB.",
    )
    parser.add_argument(
        "--parametros-produtos",
        type=Path,
        default=Path("Parametros_Produtos.xlsx"),
        help="Planilha com parâmetros de produto (peso, pallet).",
    )
    parser.add_argument(
        "--capacidade-veiculos",
        type=Path,
        default=Path("Capacidade_veiculo_por_Transportadora.xlsx"),
        help="Planilha com capacidade de veículos por transportadora.",
    )
    parser.add_argument(
        "--simular-produto",
        type=str,
        default=None,
        metavar="PRODUTO",
        help="Nome ou código do produto para simulação de carga.",
    )
    parser.add_argument(
        "--simular-destino",
        type=str,
        default=None,
        metavar="DESTINO",
        help="Destino para a simulação de carga.",
    )
    parser.add_argument(
        "--simular-quantidade",
        type=float,
        default=None,
        metavar="QTD",
        help="Quantidade de unidades a transportar na simulação.",
    )
    args = parser.parse_args()

    inbound_path = resolver_caminho(args.inbound)
    transferencia_path = resolver_caminho(args.transferencia)
    html_saida = resolver_caminho(args.output_html)
    csv_saida = resolver_caminho(args.output_csv) if str(args.output_csv).strip() else None

    consolidado, melhores = executar(
        inbound_path=inbound_path,
        transferencia_path=transferencia_path,
        html_saida=html_saida,
        csv_saida=csv_saida,
        hubs=args.hubs,
    )

    print(f"HTML gerado em: {html_saida.resolve()}")
    if csv_saida:
        print(f"CSV consolidado em: {csv_saida.resolve()}")
    print(f"Total de combinações: {len(consolidado):,}")
    print(f"Total de combinações destino+veículo com melhor rota: {len(melhores):,}")

    # ── Simulação de carga (opcional) ─────────────────────────────────────
    if args.simular_produto and args.simular_destino and args.simular_quantidade:
        prod_path = resolver_caminho(args.parametros_produtos)
        cap_path = resolver_caminho(args.capacidade_veiculos)

        df_produtos = carregar_parametros_produtos(prod_path) if prod_path.exists() else pd.DataFrame()
        df_capacidade = carregar_capacidade_veiculos(cap_path) if cap_path.exists() else None

        if df_produtos.empty:
            print(
                f"\n[AVISO] Arquivo de produtos não encontrado: {prod_path}. "
                "A simulação requer a planilha Parametros_Produtos.xlsx."
            )
        else:
            # Recarrega inbound e transferência como DataFrames para a simulação
            cols = ColunasPadrao()
            required_inbound = [
                cols.operacao, cols.destino, cols.uf_destino,
                cols.tipo_transp, cols.transportadora, cols.veiculo_kg, cols.frete,
            ]
            required_transf = [
                cols.operacao, cols.origem, cols.destino, cols.uf_destino,
                cols.tipo_transp, cols.transportadora, cols.veiculo_kg, cols.frete,
            ]
            df_inbound = ler_tabela(inbound_path, obrigatorias=required_inbound)
            df_transferencia = ler_tabela(transferencia_path, obrigatorias=required_transf)

            try:
                r = simular_carga(
                    produto=args.simular_produto,
                    destino=args.simular_destino,
                    quantidade=args.simular_quantidade,
                    df_produtos=df_produtos,
                    df_inbound=df_inbound,
                    df_transferencia=df_transferencia,
                    df_capacidade=df_capacidade,
                    hubs=args.hubs,
                )
                print("\n" + "=" * 62)
                print("  SIMULAÇÃO DE CARGA")
                print("=" * 62)
                print(f"  Produto           : {r.produto}")
                print(f"  Destino           : {r.destino}")
                print(f"  Quantidade        : {r.quantidade:,.0f} unidades")
                print(f"  Peso Total        : {r.peso_total:,.0f} kg")
                print(f"  Pallets Totais    : {r.pallets_totais}")
                print(f"  Estratégia        : {r.estrategia}")
                print(f"  Formação          : {r.formacao_veiculos}")
                print(f"  Transportadora    : {r.transportadora}")
                print(f"  Qtd. Veículos     : {r.quantidade_veiculos}")
                print(f"  Custo por Veículo : {_formatar_brl(r.custo_por_veiculo)}")
                print(f"  Custo Total       : {_formatar_brl(r.custo_total)}")
                print(f"  Cenários Testados : {r.cenarios_avaliados}")
                print("-" * 62)
                print(f"  RECOMENDAÇÃO: {r.insight}")
                print("=" * 62)
            except ValueError as exc:
                print(f"\n[SIMULAÇÃO] Erro: {exc}")



if __name__ == "__main__":
    main()
