from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse
import html

import pandas as pd
from pandas.api.types import is_numeric_dtype


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
    veiculos = sorted(df["VEICULO 1"].astype(str).unique().tolist(), key=chave_ordenacao_veiculo)
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

    body_rows: List[str] = []
    for _, row in df.iterrows():
        is_best = bool(row["MELHOR_POR_DESTINO_VEICULO"])
        row_class = "best" if is_best else "expensive"
        total_cell_class = "total-cell best-total" if is_best else "total-cell"
        veiculo_2 = "" if str(row["VEICULO 2"]) in {"", "nan"} else html.escape(str(row["VEICULO 2"]))
        body_rows.append(
            "".join(
                [
                    f'<tr class="{row_class}" data-destino="{html.escape(str(row["DESTINO"]))}" '
                    f'data-veiculo="{html.escape(str(row["VEICULO 1"]))}" '
                    f'data-total="{row["TOTAL"]:.6f}">',
                    f"<td>{html.escape(str(row['DESTINO']))}</td>",
                    f"<td>{html.escape(str(row['ROTA']))}</td>",
                    f"<td>{html.escape(str(row['HUB']))}</td>",
                    f"<td>{html.escape(str(row['TRANSP. 1']))}</td>",
                    f"<td>{html.escape(str(row['VEICULO 1']))}</td>",
                    f"<td>{_formatar_brl(float(row['CUSTO 1']))}</td>",
                    f"<td>{html.escape(str(row['TRANSP. 2']))}</td>",
                    f"<td>{veiculo_2}</td>",
                    f"<td>{_formatar_brl(float(row['CUSTO 2']))}</td>",
                    f'<td class="{total_cell_class}">{_formatar_brl(float(row["TOTAL"]))}</td>',
                    "</tr>",
                ]
            )
        )

    melhores = (
        df[df["MELHOR_POR_DESTINO_VEICULO"]]
        .sort_values(["DESTINO", "VEICULO 1", "TOTAL", "ROTA", "HUB"])
        .drop_duplicates(subset=["DESTINO", "VEICULO 1"], keep="first")
        .reset_index(drop=True)
    )

    melhores_rows: List[str] = []
    for _, row in melhores.iterrows():
        melhores_rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{html.escape(str(row['DESTINO']))}</td>",
                    f"<td>{html.escape(str(row['ROTA']))}</td>",
                    f"<td>{html.escape(str(row['HUB']))}</td>",
                    f"<td>{html.escape(str(row['TRANSP. 1']))}</td>",
                    f"<td>{html.escape(str(row['TRANSP. 2']))}</td>",
                    f"<td>{html.escape(str(row['VEICULO 1']))}</td>",
                    f"<td>{_formatar_brl(float(row['TOTAL']))}</td>",
                    "</tr>",
                ]
            )
        )

    html_content = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Dashboard de Rotas Logísticas</title>
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

    .hero {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 22px;
      animation: rise 0.5s ease;
    }}

    .hero-main {{
      display: flex;
      align-items: center;
      gap: 18px;
      min-width: 0;
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
    }}

    .hero h1 {{
      margin: 0;
      font-size: clamp(1.4rem, 3vw, 2rem);
      letter-spacing: 0.02em;
    }}

    .hero p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .chip {{
      background: linear-gradient(120deg, #173628, #1d4a37);
      color: #b7f4cf;
      border: 1px solid #2a6548;
      border-radius: 999px;
      padding: 6px 11px;
      font-size: 0.8rem;
      font-weight: 700;
    }}

    .panel {{
      background: linear-gradient(180deg, #17242e 0%, #111a21 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 18px 38px rgba(0, 0, 0, 0.25);
      animation: rise 0.6s ease;
    }}

    .filters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
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

    .table-wrap {{
      width: 100%;
      overflow: auto;
      max-height: 62vh;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1100px;
    }}

    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #1d2c38;
      white-space: nowrap;
      font-size: 0.88rem;
    }}

    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #0f1921;
      text-align: left;
      cursor: pointer;
      user-select: none;
      letter-spacing: 0.02em;
      color: #b8cedd;
    }}

    tbody tr.best {{
      background: rgba(39, 195, 106, 0.18);
      box-shadow: inset 4px 0 0 #27c36a;
    }}

    tbody tr.expensive {{
      background: rgba(214, 80, 80, 0.08);
    }}

    tbody tr.best:hover {{
      background: rgba(39, 195, 106, 0.24);
    }}

    tbody tr:hover {{
      background: rgba(72, 211, 166, 0.16);
    }}

    .total-cell {{
      font-weight: 700;
    }}

    .best-total {{
      color: #c9ffd8;
      font-weight: 800;
    }}

    .best-toggle-btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 16px;
      padding: 10px 20px;
      background: linear-gradient(135deg, #1b4d37, #163d2c);
      color: #b7f4cf;
      border: 1px solid #2a6548;
      border-radius: 999px;
      font-family: inherit;
      font-size: 0.88rem;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: 0.03em;
      transition: background 0.2s, box-shadow 0.2s, transform 0.15s;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}

    .best-toggle-btn:hover {{
      background: linear-gradient(135deg, #226046, #1c4d38);
      box-shadow: 0 6px 20px rgba(39, 195, 106, 0.25);
      transform: translateY(-1px);
    }}

    .best-toggle-btn .btn-arrow {{
      display: inline-block;
      transition: transform 0.25s;
      font-style: normal;
    }}

    .best-toggle-btn.open .btn-arrow {{
      transform: rotate(180deg);
    }}

    .best-box {{
      margin-top: 10px;
      background: linear-gradient(160deg, #10251d, #0f1a15);
      border: 1px solid #224d39;
      border-radius: 14px;
      overflow: hidden;
      max-height: 0;
      opacity: 0;
      transition: max-height 0.4s ease, opacity 0.3s ease, margin-top 0.3s ease;
    }}

    .best-box.visible {{
      max-height: 1200px;
      opacity: 1;
      margin-top: 10px;
      animation: rise 0.35s ease;
    }}

    .best-box h2 {{
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid #224d39;
      font-size: 1rem;
      color: #b8f8d3;
    }}

    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.82rem;
    }}

    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}

    @keyframes rise {{
      from {{ transform: translateY(8px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}

    @media (max-width: 800px) {{
      .wrap {{ padding: 18px 10px 28px; }}
      th, td {{ padding: 9px 8px; font-size: 0.8rem; }}
      .hero {{ align-items: stretch; flex-direction: column; }}
      .hero-main {{ align-items: flex-start; }}
      .brand-mark {{ width: 150px; height: 72px; border-radius: 16px; }}
      .brand-mark img {{ width: 100%; height: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-main">
        {logo_markup}
        <div class="hero-copy">
          <h1>Otimizador de Frete • Inbound + Transferência</h1>
          <p>Simulação de rotas DIRETO e HUB com menor custo por destino e veículo destacado automaticamente.</p>
        </div>
      </div>
      <div class="chips">
        <span class="chip">Combinações: {len(df):,}</span>
        <span class="chip">Destinos: {df['DESTINO'].nunique():,}</span>
        <span class="chip">Melhores rotas: {len(melhores):,}</span>
      </div>
    </section>

    <section class="panel">
      <div class="filters">
        <div class="control">
          <label for="destinoFilter">Destino</label>
          <select id="destinoFilter">
            {_options_html(destinos, include_all=True)}
          </select>
        </div>
        <div class="control">
          <label for="veiculoFilter">Tipo de Veículo</label>
          <select id="veiculoFilter">
            {_options_html(veiculos, include_all=True)}
          </select>
        </div>
      </div>

      <div class="table-wrap">
        <table id="rotasTable">
          <thead>
            <tr>
              <th data-col="0">DESTINO</th>
              <th data-col="1">ROTA</th>
              <th data-col="2">HUB</th>
              <th data-col="3">TRANSP. 1</th>
              <th data-col="4">VEÍCULO 1</th>
              <th data-col="5">CUSTO 1</th>
              <th data-col="6">TRANSP. 2</th>
              <th data-col="7">VEÍCULO 2</th>
              <th data-col="8">CUSTO 2</th>
              <th data-col="9">TOTAL</th>
            </tr>
          </thead>
          <tbody>
            {''.join(body_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <div class="legend">
      <span><i class="dot" style="background: var(--ok)"></i> Melhor opção por destino e veículo</span>
      <span><i class="dot" style="background: var(--bad)"></i> Opções mais caras</span>
    </div>

    <button class="best-toggle-btn" id="bestToggleBtn" onclick="toggleBestBox()">
      <span>Ver melhor rota por destino e veículo</span>
      <i class="btn-arrow">&#9660;</i>
    </button>

    <section class="best-box" id="bestBox">
      <h2>Melhor rota por destino e veículo</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>DESTINO</th>
              <th>ROTA</th>
              <th>HUB</th>
              <th>TRANSP. 1</th>
              <th>TRANSP. 2</th>
              <th>VEÍCULO</th>
              <th>TOTAL</th>
            </tr>
          </thead>
          <tbody>
            {''.join(melhores_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    const destinoFilter = document.getElementById('destinoFilter');
    const veiculoFilter = document.getElementById('veiculoFilter');
    const table = document.getElementById('rotasTable');
    const tbody = table.querySelector('tbody');

    function applyFilters() {{
      const destino = destinoFilter.value;
      const veiculo = veiculoFilter.value;
      const rows = tbody.querySelectorAll('tr');

      rows.forEach(row => {{
        const okDestino = !destino || row.dataset.destino === destino;
        const okVeiculo = !veiculo || row.dataset.veiculo === veiculo;
        row.style.display = (okDestino && okVeiculo) ? '' : 'none';
      }});
    }}

    destinoFilter.addEventListener('change', applyFilters);
    veiculoFilter.addEventListener('change', applyFilters);

    let sortDir = 1;
    function parseCurrencyToNumber(text) {{
      return Number(String(text).replace('R$', '').replaceAll('.', '').replace(',', '.').trim()) || 0;
    }}

    function compareVehicle(at, bt) {{
      const an = Number(at);
      const bn = Number(bt);
      const aIsNum = at !== '' && !Number.isNaN(an);
      const bIsNum = bt !== '' && !Number.isNaN(bn);

      if (aIsNum && bIsNum) {{
        return an - bn;
      }}

      if (aIsNum !== bIsNum) {{
        return aIsNum ? -1 : 1;
      }}

      return at.localeCompare(bt, 'pt-BR');
    }}

    function toggleBestBox() {{
      const box = document.getElementById('bestBox');
      const btn = document.getElementById('bestToggleBtn');
      const isOpen = box.classList.contains('visible');

      if (isOpen) {{
        box.classList.remove('visible');
        btn.classList.remove('open');
        btn.querySelector('span').textContent = 'Ver melhor rota por destino e veículo';
      }} else {{
        box.classList.add('visible');
        btn.classList.add('open');
        btn.querySelector('span').textContent = 'Fechar melhor rota por destino e veículo';
      }}
    }}

    table.querySelectorAll('th').forEach((th, idx) => {{
      th.addEventListener('click', () => {{
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {{
          const at = a.children[idx].textContent.trim();
          const bt = b.children[idx].textContent.trim();

          if (idx === 4 || idx === 7) {{
            return compareVehicle(at, bt) * sortDir;
          }}

          if (idx === 5 || idx === 8 || idx === 9) {{
            return (parseCurrencyToNumber(at) - parseCurrencyToNumber(bt)) * sortDir;
          }}

          return at.localeCompare(bt, 'pt-BR') * sortDir;
        }});

        rows.forEach(r => tbody.appendChild(r));
        sortDir *= -1;
        applyFilters();
      }});
    }});
  </script>
</body>
</html>
"""

    destino_saida.write_text(html_content, encoding="utf-8")


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
    args = parser.parse_args()

    csv_saida = args.output_csv if str(args.output_csv).strip() else None

    consolidado, melhores = executar(
        inbound_path=args.inbound,
        transferencia_path=args.transferencia,
        html_saida=args.output_html,
        csv_saida=csv_saida,
        hubs=args.hubs,
    )

    print(f"HTML gerado em: {args.output_html.resolve()}")
    if csv_saida:
        print(f"CSV consolidado em: {csv_saida.resolve()}")
    print(f"Total de combinações: {len(consolidado):,}")
    print(f"Total de combinações destino+veículo com melhor rota: {len(melhores):,}")


if __name__ == "__main__":
    main()
