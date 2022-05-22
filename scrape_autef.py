"""
BR - Pará State - Wood Pulp
Scrapping Timber Authorization PDF documents
"Script took: 2:17:07.490039 to run." time to run it at the first time fully
"""


import os
from datetime import datetime
from uuid import uuid4
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import tabula as tb

from subprocess import CalledProcessError

your_user_name = ""

def main():
    start_time = datetime.now()
    print("Script started at:", start_time)

    input_dir = f".\\pdfs\\initial\\"
    rot_dir = f".\\pdfs\\rotated\\"
    output_dir = f".\\pdfs\\scraped\\"

    autef = Autef(input_dir, rot_dir, output_dir)
    autef.run()

    print("Script took:", datetime.now() - start_time, "to run.")


class Autef:
    def __init__(self, files_dir, rotate_dir, out_dir):
        self.files_dir = files_dir
        self.rotate_dir = rotate_dir
        self.out_dir = out_dir

    def _create_file_list(self):
        infos = []
        motivo = []

        for i, j in zip(glob.glob(f'{self.files_dir}*.pdf'), glob.glob(f"{self.rotate_dir}*.pdf")):
            infos.append(i)
            motivo.append(j)

        return infos, motivo

    def _scrapping_each_autef(self):
        def _info(i):
            inf = tb.read_pdf_with_template(i, "pa_autef_info.tabula-template.json", silent=True,
                                            lattice=True, multiple_tables=True,
                                            pandas_options={"header": None})
            inf = pd.concat(inf)

            return inf

        def _info_header():
            header = _info(i).iloc[:2, 0:1]
            header[0] = header[0].str.split("\r").str.join(", ")
            header["pdf_name"] = os.path.basename(i)
            header["autef_num"] = header[0][header[0].str.contains("AUTEF")].str.extract("(\d+/+\d*)")
            header["validade_ate"] = header[0][header[0].str.contains("VALIDADE")].str.extract("(\d+/+\d*/+\d*)")
            header = header[["pdf_name", "autef_num", "validade_ate"]].iloc[0, :]
            header = header.reset_index().transpose()
            header.columns = header.iloc[0]
            header = header.iloc[1:, :]
            header = header[["pdf_name", "autef_num", "validade_ate"]]
            header = header.reindex().sort_index()
            print(os.path.basename(i))
            return header

        def _protocol_car():
            p_car = _info(i).iloc[2:4, 0:1]
            p_car[0] = p_car[0].str.split("\r").str.join(", ")
            p_car["protocolo_num"] = p_car[0].astype(str).str.split(',').str[0]
            p_car["protocolo_data"] = p_car[0].astype(str).str.split(',').str[1]
            p_car["car_num"] = \
                p_car[p_car["protocolo_num"].str.contains("Cadastro Ambiental Rural")]["protocolo_num"]
            p_car["licenca_num"] = \
                p_car[p_car["protocolo_data"].str.contains("Licença Atividade Rural", na=False)] \
                    ["protocolo_data"]
            p_car = p_car.iloc[0].reset_index().transpose()
            p_car.columns = p_car.iloc[0]
            p_car = p_car.iloc[1:, :]
            p_car["protocolo_num"] = p_car["protocolo_num"].str.extract("(\d+/+\d*)")
            p_car["protocolo_data"] = p_car["protocolo_data"].str.extract("(\d+/+\d*/+\d*)")
            p_car["car_num"] = p_car["car_num"].str.extract("(\d+/+\d*)")
            p_car["licenca_num"] = p_car["licenca_num"].str.extract("(\d+/+\d*)")
            protocol_car = p_car[["protocolo_num", "protocolo_data", "car_num", "licenca_num"]]
            protocol_car = protocol_car.reindex().sort_index()

            return protocol_car

        def _respons_tecnico():
            resp_tecnico = _info(i).iloc[4:6, 0:2]
            resp_tecnico["responsavel_tecnico"] = resp_tecnico.iloc[1]
            resp_tecnico["crea_num"] = resp_tecnico.iloc[1, 1]
            resp_tecnico = resp_tecnico.iloc[0].reset_index().transpose()
            resp_tecnico.columns = resp_tecnico.iloc[0]
            resp_tecnico = resp_tecnico.iloc[1:]
            resp_tecnico = resp_tecnico[["responsavel_tecnico", "crea_num"]]
            resp_tecnico["crea_num"] = resp_tecnico["crea_num"].str.split("CREA:").str[1]
            resp_tecnico = resp_tecnico.reindex().sort_index()

            return resp_tecnico

        def _propriedade():
            def extract_pdf_info():
                # TODO make astype.split function
                prop = _info(i).iloc[7:13, :2].copy()
                prop[0] = prop[0].str.split("\r").str.join(", ")
                prop[1] = prop[1].str.split("\r").str.join(", ")
                prop["proprietario_nome"] = \
                    prop[0][prop[0].str.contains("PROPRIETÁRIO")].astype(str).str.split(',').str[0].str.split(
                        "PROPRIETÁRIO:").str[1]
                prop["proprietario_cnpj"] = \
                    prop[0][prop[0].str.contains("PROPRIETÁRIO")].astype(str).str.split(',').str[1].str.split(
                        "CPF/CNPJ:").str[1]
                prop["detentor_nome"] = \
                    prop[0][prop[0].str.contains("DETENTOR")].astype(str).str.split(',').str[0].str.split(
                        "DETENTOR:").str[1]
                prop["detentor_cnpj"] = \
                    prop[0][prop[0].str.contains("DETENTOR")].astype(str).str.split(',').str[0].str.split(
                        "DETENTOR:").str[1]
                prop["municipio"] = \
                    prop[0][prop[0].str.contains("MUNICÍPIO")].astype(str).str.split(',').str[1].str.split(
                        "MUNICÍPIO:").str[1]
                try:
                    prop.loc[prop[0].str.contains("COORDENADAS", na=False), "municipio"] = \
                        prop[0].astype(str).str.split(',').str[1].str.split("COORDENADAS").str[0].str.split(":").str[1]

                except ValueError:
                    prop["municipio"] = \
                        prop[0][prop[0].str.contains("MUNICÍPIO")].astype(str).str.split(',').str[1].str.split(
                            "MUNICÍPIO:").str[1]

                prop["geo_info"] = \
                prop[0][prop[0].str.contains("MUNICÍPIO")].astype(str).str.split('COORDENADAS GEOGRÁFICAS:').str[1]

                # extract datum info
                sad = prop["geo_info"].str.extract("(SAD69)")
                sad_not_nan = ~prop["geo_info"].str.extract("(SAD69)").isna()

                sirgas = prop["geo_info"].str.extract("(SIRGAS2000)")
                sirgas_not_nan = ~prop["geo_info"].str.extract("(SIRGAS2000)").isna()

                wgs = prop["geo_info"].str.extract("(WGS84)")
                wgs_not_nan = ~prop["geo_info"].str.extract("(WGS84)").isna()

                conditions = [sad_not_nan, sirgas_not_nan, wgs_not_nan]
                values = [sad, sirgas, wgs]
                prop["datum"] = np.select(conditions, values, default=np.nan)

                prop["porte"] = prop["geo_info"].str.split("PORTE:").str[-1]

                # extract coordinate info
                try:
                    prop["coord_"] = prop[prop["geo_info"].str.contains("FUSO")]["geo_info"] \
                        .str.split("HEMISFERIO: Sul - ").str[1].str.split("FUSO: [0-9]* - ", regex=True) \
                        .str[1].str.split("PORTE: [a-zA-Z]*\w* - [IÇ]*", regex=True).str[0]
                except ValueError:
                    prop["coord_"] = ""

                try:
                    prop["coord"] = \
                        prop["geo_info"].str.split("HEMISFERIO: Sul - ").str[1].str.split("FUSO: [0-9]* - ", regex=True) \
                            .str[0].str.split("PORTE: [a-zA-Z]*\w* - [IÇ]*", regex=True).str[0]
                except AttributeError:
                    prop["coord"] = ""

                prop.loc[((prop["coord"] == "") & (~prop["coord_"].isna())), "coord"] = prop["coord_"]
                try:
                    prop.loc[~prop["coord"].isna(), "coord"] = prop["coord"].str.rstrip(", ")
                except:
                    pass

                if "coord_" in prop.columns:
                    del prop["coord_"]

                del prop["geo_info"]

                prop["imovel"] = \
                    prop[0][prop[0].str.contains("IMÓVEL")].astype(str).str.split(',').str[0].str.split(
                        "IMÓVEL:").str[1]
                prop["imovel_area"] = \
                    prop[0][prop[0].str.contains("Área Total da propriedade")].astype(str).str.split(
                        ':,').str[1].str.split("ha") \
                        .str[0].str.strip()
                prop["reserva_legal_area"] = \
                    prop[1][prop[1].str.startswith("Área de Reserva Legal", na=False)].astype(str).str.split(
                        ':,').str[1] \
                        .str.split("ha").str[0].str.strip()
                prop["mfs_area"] = \
                    prop[0][prop[0].str.contains("MFS")].astype(str).str.split(':,').str[1].str.split(
                        "ha").str[0].str.strip()
                prop["antropizada_area"] = \
                    prop[1][prop[1].str.contains("Antropizada", na=False)].astype(str).str.split(':,').str[
                        1].str.split("ha").str[0] \
                        .str.strip()
                prop["app_upa"] = \
                    prop[0][prop[0].str.contains("APP da UPA")].astype(str).str.split(":,").str[1].str.split(
                        "ha").str[0].str.strip()
                prop["area_autorizada"] = \
                    prop[1][prop[1].str.startswith("Área Autorizada", na=False)].astype(str).str.split(
                        ":,").str[1].str.split("ha").str[0] \
                        .str.strip()

                return prop

            def adjust_values():
                prop = extract_pdf_info()
                prop.loc[
                    prop["antropizada_area"].str.contains("XXXX", na=False), "antropizada_area"] = np.NaN
                prop.loc[prop["app_upa"].str.contains("XXXX", na=False), "app_upa"] = np.NaN

                return prop

            def organize_df():
                prop = adjust_values()
                prop = prop.iloc[:, 2:]
                prop["detentor_nome"] = prop["detentor_nome"].fillna(method="backfill")
                prop["detentor_cnpj"] = prop["detentor_cnpj"].fillna(method="backfill")
                prop["municipio"] = prop["municipio"].fillna(method="backfill")
                prop["imovel"] = prop["imovel"].fillna(method="backfill")
                prop["datum"] = prop["datum"].fillna(method="backfill")
                prop["porte"] = prop["porte"].fillna(method="backfill")
                prop["coord"] = prop["coord"].fillna(method="backfill")
                prop["imovel_area"] = prop["imovel_area"].fillna(method="backfill")
                prop["reserva_legal_area"] = prop["reserva_legal_area"].fillna(method="backfill")
                prop["mfs_area"] = prop["mfs_area"].fillna(method="backfill")
                prop["antropizada_area"] = prop["antropizada_area"].fillna(method="backfill")
                prop["app_upa"] = prop["app_upa"].fillna(method="backfill")
                prop["area_autorizada"] = prop["area_autorizada"].fillna(method="backfill")
                prop = prop.iloc[0].reset_index().transpose()
                prop.columns = prop.iloc[0]
                prop = prop.iloc[1:]
                prop = prop.reset_index(drop=True)

                return prop

            return organize_df()

        def _especies(i):
            global diverso
            ssp = tb.read_pdf_with_template(i, "pa_autef_ssp.tabula-template.json", silent=True,
                                            lattice=True, multiple_tables=True,
                                            pandas_options={"header": None},
                                            )

            try:
                ssp = pd.concat(ssp)
                if ssp.shape[0] >= 6:
                    ssp.columns = ssp.iloc[6]
                    ssp = ssp.iloc[7:].copy()
                else:
                    pass

            except:
                if not type(ssp) == list:
                    if ssp.shape[0] <= 4:
                        ssp = tb.read_pdf_with_template(i, "pa_autef_spp_longer.tabula-template.json", silent=True,
                                                        lattice=True, multiple_tables=True,
                                                        pandas_options={"header": None},
                                                        )
                        if ssp:
                            ssp = pd.concat(ssp)
                            ssp.columns = ssp.iloc[0]
                            ssp = ssp.iloc[1:].copy()
                        else:
                            ssp = pd.DataFrame({"quantidade_m3": [np.NaN], "quantidade_ha": [np.NaN],
                                                "nome_popular": [np.NaN], "nome_cientifico": [np.NaN],
                                                "individuos": [np.NaN]})
                else:
                    ssp = pd.DataFrame({"quantidade_m3": [np.NaN], "quantidade_ha": [np.NaN],
                                        "nome_popular": [np.NaN], "nome_cientifico": [np.NaN],
                                        "individuos": [np.NaN]})

            try:
                diverso = ssp[ssp["NOME CIENTÍFICO"].str.contains("Diversos")]
            except:
                pass

            greater = ssp.shape[0] > 2
            less = ssp.shape[0] <= 2

            if "ESPÉCIES FLORESTAIS DO POA" in ssp.columns:
                ssp.columns = ssp.iloc[0]
                ssp = ssp.iloc[1:].copy()
            elif "NOME CIENTÍFICO" in ssp.columns:
                try:
                    diverso = ssp[ssp["NOME CIENTÍFICO"].str.contains("Diversos")]
                except:
                    pass

                if all(diverso) & greater:
                    if ssp.shape[0] >= 1:
                        if "NOME CIENTÍFICO" in ssp.columns:
                            ssp.columns = ssp.columns
                            ssp = ssp.iloc[1:].copy()
                        elif ssp.iloc[3][0] == "ESPÉCIES FLORESTAIS DO POA":
                            ssp.columns = ssp.iloc[4]
                            ssp = ssp.iloc[5:].copy()
                    else:
                        ssp = ssp.iloc[5:].copy()
                elif all(diverso) & less:
                    ssp = ssp
            else:
                ssp.columns = ssp.iloc[0]
                ssp = ssp.iloc[1:].copy()

            ssp = ssp.rename(columns={"NOME CIENTÍFICO": "nome_cientifico", "NOME POPULAR": "nome_popular",
                                      "Indivíduos": "individuos",
                                      "por ha": "quantidade_ha", "TOTAL": "quantidade_m3"})
            try:
                if "individuos" in ssp.columns:
                    ssp["quantidade_m3"].iloc[-1] = ssp["quantidade_ha"].iloc[-1]
                    ssp["quantidade_ha"].iloc[-1] = ssp["individuos"].iloc[-1]
                    ssp["individuos"].iloc[-1] = ssp["nome_popular"].iloc[-1]
                    ssp["nome_popular"].iloc[-1] = ssp["nome_cientifico"].iloc[-1]
                elif all(diverso) & less:
                    ssp["nome_cientifico"] = ssp["nome_cientifico"].iloc[0]
                    ssp["nome_popular"] = ssp["nome_cientifico"].iloc[0]
                    ssp["individuos"] = np.NaN
                    ssp["quantidade_ha"] = ssp["quantidade_ha"].iloc[0]
                    ssp["quantidade_m3"] = ssp["quantidade_m3"].iloc[0]
                else:
                    ssp["quantidade_m3"].iloc[-1] = ssp["quantidade_ha"].iloc[-1]
                    ssp["quantidade_ha"].iloc[-1] = ssp["nome_popular"].iloc[-1]
                    ssp["nome_popular"].iloc[-1] = ssp["nome_cientifico"].iloc[-1]
                    ssp["individuos"] = np.NaN
            except KeyError:
                ssp = pd.DataFrame({"quantidade_m3": [np.NaN], "quantidade_ha": [np.NaN],
                                    "nome_popular": [np.NaN], "nome_cientifico": [np.NaN],
                                    "individuos": [np.NaN]})
            except TypeError:
                ssp = pd.DataFrame({"quantidade_m3": [np.NaN], "quantidade_ha": [np.NaN],
                                    "nome_popular": [np.NaN], "nome_cientifico": [np.NaN],
                                    "individuos": [np.NaN]})

            ssp = ssp.reindex().sort_index()

            return ssp

        def _load_lateral_cel(j):
            try:
                lat_cel = tb.read_pdf_with_template(j, "pa_autef_lateral_cel.tabula-template.json", silent=True,
                                                    stream=True)
                lat_cel = pd.concat(lat_cel)

            except CalledProcessError:
                lat_cel = pd.DataFrame({"motivo": ["Unsolved"]})
            except AttributeError:
                lat_cel = pd.DataFrame({"motivo": ["Unsolved"]})

            return lat_cel

        def _extract_lateral_cel(j):
            global df
            data = _load_lateral_cel(j)

            if data.shape[0] == 0:
                data = data.columns.str.split("em:")[0]
                if len(data) == 2:
                    data = data[1].strip()
                    df = pd.DataFrame(data=[data], columns=['motivo'])

            elif data.shape[0] >= 0:
                # data.columns = ["motivo"]
                data = data.iloc[0]
                data = data.str.split(";").str[1]
                df = pd.DataFrame(data=[data], columns=["motivo"])
                try:
                    df["motivo"] = df["motivo"].str.replace("ncido em:", "Vencido em:")
                    df["motivo"] = df["motivo"].str.replace("do por falha na e", "Falha na elaboração")
                    df["motivo"] = df["motivo"].str.replace("nto: Cancelado", "Cancelado")
                    df["motivo"] = df["motivo"].str.replace(":", "")
                except:
                    df["motivo"] = np.nan

            elif "motivo" not in data.columns:
                data = data.columns[2]
                data = data.split(",")
                col = "motivo"
                col = col.split(":")[0]
                col = col[:].split(",")
                col_data = dict(zip(col, data))
                df = pd.DataFrame(col_data, index=[0])
            else:
                df = data

            return df

        def _concatenate(header, protoc_car, resp_tecnico, propr, lat_cel, esp):

            # split species list from the total exploited
            try:
                ssp_overall = esp.iloc[-1].copy()
            except:
                ssp_overall = esp
            ssp_overall["index"] = 0
            ssp_overall = ssp_overall[["index", "individuos", "quantidade_ha", "quantidade_m3"]]
            ssp_overall = ssp_overall.reset_index().transpose()
            ssp_overall.columns = ssp_overall.iloc[0]
            ssp_overall = ssp_overall.iloc[1:]
            ssp_overall = ssp_overall.reset_index(drop=True)

            df = pd.concat(
                [header, protoc_car, resp_tecnico, propr, lat_cel, ssp_overall, esp], axis=1).sort_index()

            del df["index"]
            df["id"] = uuid4()

            ssp_detailed = df.iloc[1:-1].copy()
            ssp_detailed = ssp_detailed[["id", "nome_cientifico", 'nome_popular', 'individuos', 'quantidade_ha']]
            ssp_detailed.columns = ["id", "nome_cientifico", "nome_popular", "ind_nan", "individuos", "quant_nan",
                                    "quantidade_ha"]
            ssp_detailed = ssp_detailed[["id", "nome_cientifico", "nome_popular", "individuos", "quantidade_ha"]]

            common_cols = ['pdf_name', 'autef_num', 'validade_ate', 'protocolo_num',
                           'protocolo_data', 'car_num', 'licenca_num', 'responsavel_tecnico',
                           'crea_num', 'proprietario_nome', 'proprietario_cnpj', 'detentor_nome',
                           'detentor_cnpj', 'municipio', 'datum', 'porte', 'coord',
                           'imovel', 'imovel_area', 'reserva_legal_area', 'mfs_area',
                           'antropizada_area', 'app_upa', 'area_autorizada', 'motivo', 'individuos',
                           'quantidade_ha', 'quantidade_m3', 'nome_cientifico', 'nome_popular']

            if df.shape[1] == 34:
                df.columns = [*common_cols, 'indi_nan', 'quant_nan', 'quantidade_t_nan', 'id']
            elif df.shape[1] == 35:
                df.columns = [*common_cols, 'indi_nan', 'quant_nan', 'quantidade_t_nan', 'nan', 'id']
            else:
                df.columns = [*common_cols, 'quant_nan', 'quantidade_t_nan', 'nan', 'nan', 'nan', 'nan', 'ind_nan',
                              'id']

            df = df[['id', *common_cols[:-2]]]

            df = df.iloc[0].reset_index().transpose()
            df.columns = df.iloc[0]
            df = df.iloc[[1]]

            return df, ssp_detailed

        info_concat = []
        ssp_concat = []

        for i, j in tqdm(zip(self._create_file_list()[0], self._create_file_list()[1]), desc="Scrapping PDF",
                         total=len(self._create_file_list()[0])):
            info_header = _info_header()
            protocol_car = _protocol_car()
            respons_tecnico = _respons_tecnico()
            propriedade = _propriedade()
            especies = _especies(i)
            lateral_cel = _extract_lateral_cel(j)

            info, ssp = _concatenate(info_header, protocol_car, respons_tecnico, propriedade, lateral_cel, especies)
            info_concat.append(info)
            ssp_concat.append(ssp)

        info = pd.concat(info_concat)
        ssp = pd.concat(ssp_concat)

        return info, ssp

    def run(self):
        def _create_out_dir():
            from pathlib import Path
            return Path(f"{self.out_dir}").mkdir(parents=True, exist_ok=True)

        _create_out_dir()

        info, ssp = self._scrapping_each_autef()

        info.to_csv(f"{self.out_dir}autef_infos_second_round.csv", sep=";", encoding="utf-8", index=False)
        ssp.to_csv(f"{self.out_dir}autef_ssp_detailed_second_round.csv", sep=";", encoding="utf-8", index=False)


if __name__ == "__main__":
    main()