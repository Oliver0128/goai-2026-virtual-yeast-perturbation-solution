from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


QC_LABEL = "Quality Control"
CONTROL_LABELS = {"DMSO", "Water"}
DESCRIPTOR_NAMES = ["mol_wt", "mol_logp", "tpsa", "hbd", "hba"]
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_metadata_entities(train_metadata: Path, test_metadata: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    frames = [pd.read_csv(train_metadata), pd.read_csv(test_metadata)]
    required = {"perturbation_no_concentration", "split_final"}
    for frame in frames:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    metadata = pd.concat(frames, ignore_index=True)
    metadata["perturbation_no_concentration"] = metadata["perturbation_no_concentration"].astype(str)
    metadata["split_final"] = metadata["split_final"].astype(str)
    metadata = metadata.loc[metadata["perturbation_no_concentration"].ne(QC_LABEL)]
    memberships: dict[str, set[str]] = {}
    roles: dict[str, str] = {}
    for name, group in metadata.groupby("perturbation_no_concentration", sort=True):
        memberships[str(name)] = set(group["split_final"].unique())
        roles[str(name)] = "control" if name in CONTROL_LABELS else "treatment"
    return memberships, roles


def load_pubchem_properties(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    if not rows:
        raise ValueError("PubChem response contains no properties")
    by_cid = {int(row["CID"]): row for row in rows}
    if len(by_cid) != len(rows):
        raise ValueError("PubChem response contains duplicate CIDs")
    return by_cid


def canonical_isomeric_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def select_feature_molecule(mol: Chem.Mol, policy: str) -> Chem.Mol:
    if policy == "full-record":
        selected = Chem.Mol(mol)
    elif policy == "fragment-parent":
        selected = rdMolStandardize.FragmentParent(mol)
    else:
        raise ValueError(f"Unknown feature molecule policy: {policy}")
    Chem.SanitizeMol(selected)
    Chem.GetSymmSSSR(selected)
    return selected


def descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    values = np.asarray(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("RDKit descriptors must be finite")
    return values


def split_flags(splits: set[str]) -> dict[str, Any]:
    train = sorted(split for split in splits if split == "train")
    validation = sorted(split for split in splits if split.startswith("val_"))
    test = sorted(split for split in splits if split.startswith("test_"))
    return {
        "seen_in_train_metadata": bool(train),
        "seen_in_validation_metadata": bool(validation),
        "seen_in_test_metadata": bool(test),
        "train_splits": ";".join(train),
        "validation_splits": ";".join(validation),
        "test_splits": ";".join(test),
    }


def build_contract(
    identifiers_path: Path,
    pubchem_path: Path,
    official_rule_document: Path,
    train_metadata: Path,
    test_metadata: Path,
    contract_output: Path,
    fingerprint_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    identifiers = pd.read_csv(identifiers_path, dtype={"pubchem_cid": "int64"})
    required_columns = {
        "competition_name",
        "pubchem_query",
        "pubchem_cid",
        "resolution",
        "feature_molecule_policy",
        "resolution_note",
    }
    missing = required_columns.difference(identifiers.columns)
    if missing:
        raise ValueError(f"Identifier map is missing columns: {sorted(missing)}")
    if identifiers["competition_name"].duplicated().any():
        raise ValueError("Competition names must be unique")
    if identifiers["pubchem_cid"].duplicated().any():
        raise ValueError("PubChem CIDs must be unique")

    memberships, roles = load_metadata_entities(train_metadata, test_metadata)
    identifier_names = set(identifiers["competition_name"])
    metadata_names = set(memberships)
    if identifier_names != metadata_names:
        raise ValueError(
            f"Identifier/metadata entity mismatch: missing={sorted(metadata_names - identifier_names)}, "
            f"extra={sorted(identifier_names - metadata_names)}"
        )

    pubchem = load_pubchem_properties(pubchem_path)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=True,
        useBondTypes=True,
        countSimulation=False,
    )
    output_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[np.ndarray] = []
    descriptor_rows: list[np.ndarray] = []

    for identifier in identifiers.sort_values("competition_name").to_dict("records"):
        name = str(identifier["competition_name"])
        cid = int(identifier["pubchem_cid"])
        if cid not in pubchem:
            raise ValueError(f"CID {cid} for {name} is absent from the PubChem response")
        source = pubchem[cid]
        source_smiles = source.get("SMILES") or source.get("IsomericSMILES")
        if not source_smiles:
            raise ValueError(f"CID {cid} has no isomeric SMILES")
        full_mol = Chem.MolFromSmiles(str(source_smiles))
        if full_mol is None:
            raise ValueError(f"RDKit cannot parse CID {cid} SMILES")
        feature_mol = select_feature_molecule(full_mol, str(identifier["feature_molecule_policy"]))
        if feature_mol is None or feature_mol.GetNumAtoms() == 0:
            raise ValueError(f"Feature molecule is empty for {name}")
        bit_vector = generator.GetFingerprint(feature_mol)
        bit_array = np.zeros((MORGAN_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vector, bit_array)
        fingerprint_rows.append(bit_array)
        descriptor_rows.append(descriptor_vector(feature_mol))
        fragments = Chem.GetMolFrags(full_mol)
        output_rows.append(
            {
                **identifier,
                "entity_role": roles[name],
                **split_flags(memberships[name]),
                "pubchem_title": source.get("Title", ""),
                "pubchem_iupac_name": source.get("IUPACName", ""),
                "pubchem_molecular_formula": source.get("MolecularFormula", ""),
                "pubchem_molecular_weight": source.get("MolecularWeight", ""),
                "pubchem_isomeric_smiles": source_smiles,
                "pubchem_connectivity_smiles": source.get("ConnectivitySMILES", ""),
                "pubchem_inchi": source.get("InChI", ""),
                "pubchem_inchikey": source.get("InChIKey", ""),
                "rdkit_full_canonical_isomeric_smiles": canonical_isomeric_smiles(full_mol),
                "rdkit_feature_canonical_isomeric_smiles": canonical_isomeric_smiles(feature_mol),
                "full_record_fragment_count": len(fragments),
                "feature_atom_count": feature_mol.GetNumAtoms(),
                "feature_morgan_on_bits": int(bit_array.sum()),
            }
        )

    contract = pd.DataFrame(output_rows)
    fingerprints = np.vstack(fingerprint_rows)
    descriptors = np.vstack(descriptor_rows)
    if fingerprints.shape != (len(contract), MORGAN_BITS):
        raise ValueError("Morgan fingerprint shape mismatch")
    if descriptors.shape != (len(contract), len(DESCRIPTOR_NAMES)):
        raise ValueError("Descriptor shape mismatch")
    if not np.isfinite(descriptors).all():
        raise ValueError("Descriptor matrix contains non-finite values")

    contract_output.parent.mkdir(parents=True, exist_ok=True)
    contract.to_csv(contract_output, index=False, lineterminator="\n")
    fingerprint_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fingerprint_output,
        competition_names=contract["competition_name"].to_numpy(dtype=str),
        pubchem_cids=contract["pubchem_cid"].to_numpy(dtype=np.int64),
        morgan_bits=fingerprints,
        descriptor_names=np.asarray(DESCRIPTOR_NAMES, dtype=str),
        descriptors_raw=descriptors,
    )

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "purpose": "Public compound-structure feature contract for GOAI OOD entity encoding",
        "official_rule_evidence": {
            "document": "GOAI Track 3 handbook revision comparison, 2026-08-14 local edition",
            "pages": [9, 10],
            "rule": "Public external data may construct compound molecular-structure features; sources and versions must be disclosed.",
            "document_sha256": sha256_file(official_rule_document),
        },
        "source": {
            "database": "PubChem",
            "interface": "PUG-REST",
            "access_date": "2026-08-16",
            "version_basis": "PubChem returned no release identifier; this snapshot is frozen by access date and raw-response SHA-256.",
            "request": "compound/cid/<56 verified CIDs>/property/Title,IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,ExactMass,MonoisotopicMass,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,XLogP,HeavyAtomCount,Charge/JSON",
            "citation_guidelines": "https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines",
            "pug_rest_documentation": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            "retrieval_tool": "Codex pubchem_database PUG-REST wrapper",
            "retrieval_tool_sha256": "6936a39131c473aa2bd845f0dfc76fed78c69ca982c5184a3e0b80930a184454",
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
        },
        "data_contract": {
            "entity_count": len(contract),
            "quality_control_excluded": True,
            "quality_control_label": QC_LABEL,
            "metadata_identity_only": True,
            "proteome_truth_loaded": False,
            "learned_statistics": False,
            "descriptor_scaling": "none; any scaling must be fit on split_final=train in the downstream model",
            "seen_in_train_count": int(contract["seen_in_train_metadata"].sum()),
            "seen_in_validation_count": int(contract["seen_in_validation_metadata"].sum()),
            "seen_in_test_count": int(contract["seen_in_test_metadata"].sum()),
            "curated_alias_count": int(contract["resolution"].eq("curated-alias").sum()),
        },
        "feature_contract": {
            "source_structure": "PubChem isomeric SMILES",
            "canonicalization": "RDKit canonical isomeric SMILES",
            "default_feature_molecule": "RDKit FragmentParent",
            "full_record_exceptions": ["Cisplatin", "NaCl"],
            "morgan": {
                "radius": MORGAN_RADIUS,
                "diameter": MORGAN_RADIUS * 2,
                "fp_size": MORGAN_BITS,
                "include_chirality": True,
                "use_bond_types": True,
                "count_simulation": False,
                "dtype": "uint8",
            },
            "raw_descriptors": DESCRIPTOR_NAMES,
        },
        "known_identity_caveats": [
            "1-10 Phenanthroline monohydrate is represented by the anhydrous 1,10-phenanthroline parent.",
            "Oligomycin is represented by Oligomycin A.",
            "Tunicamycin is represented by Tunicamycin A; the family/mixture identity is not uniquely determined by metadata.",
        ],
        "hashes": {
            "identifiers_sha256": sha256_file(identifiers_path),
            "pubchem_response_sha256": sha256_file(pubchem_path),
            "train_metadata_sha256": sha256_file(train_metadata),
            "test_metadata_sha256": sha256_file(test_metadata),
            "contract_sha256": sha256_file(contract_output),
            "fingerprints_sha256": sha256_file(fingerprint_output),
        },
    }
    write_json(manifest_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the audited GOAI compound structure contract")
    parser.add_argument("--identifiers", type=Path, required=True)
    parser.add_argument("--pubchem-properties", type=Path, required=True)
    parser.add_argument("--official-rule-document", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--fingerprint-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_contract(
        identifiers_path=args.identifiers,
        pubchem_path=args.pubchem_properties,
        official_rule_document=args.official_rule_document,
        train_metadata=args.train_metadata,
        test_metadata=args.test_metadata,
        contract_output=args.contract_output,
        fingerprint_output=args.fingerprint_output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
