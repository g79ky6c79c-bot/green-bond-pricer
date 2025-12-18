import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

###############################################################################
# 1. Core Data Structures
###############################################################################


@dataclass
class DiscountCurve:
    """
    Simple deterministic zero-coupon curve.
    Times in years, zero_rates as annual continuously-compounded rates or simple
    (depending on convention used in discount_factor()).
    """
    times: List[float]
    zero_rates: List[float]

    def _interp_rate(self, t: float) -> float:
        """Linear interpolation of zero rates in maturity space."""
        if t <= self.times[0]:
            return self.zero_rates[0]
        if t >= self.times[-1]:
            return self.zero_rates[-1]
        for i in range(1, len(self.times)):
            if t <= self.times[i]:
                t0, t1 = self.times[i - 1], self.times[i]
                r0, r1 = self.zero_rates[i - 1], self.zero_rates[i]
                w = (t - t0) / (t1 - t0)
                return r0 + w * (r1 - r0)
        return self.zero_rates[-1]

    def discount_factor(self, t: float, comp: str = "cont") -> float:
        """
        Discount factor for maturity t.
        comp = "cont": continuously-compounded zero rates: DF = exp(-r * t)
        comp = "simple": simple annual compounding: DF = 1 / (1 + r)^t
        """
        r = self._interp_rate(t)
        if comp == "cont":
            return math.exp(-r * t)
        elif comp == "simple":
            return 1.0 / ((1.0 + r) ** t)
        else:
            raise ValueError("Unsupported compounding convention.")


@dataclass
class Bond:
    face_value: float
    coupon_rate: float        # annual coupon rate, e.g. 0.02 for 2%
    payment_frequency: int    # e.g. 1, 2, 4
    maturity_years: float
    issue_price: float        # price at issuance (clean)
    credit_spread: float      # constant credit spread (in absolute terms, e.g. 0.01 for 100 bps)


###############################################################################
# 2. Fixed Income Engine for Green Bond
###############################################################################


def generate_cashflows(bond: Bond) -> Tuple[List[float], List[float]]:
    """
    Generate payment times and cashflows for a standard fixed-rate bullet bond.
    Assumes:
    - level coupon
    - payment in arrears
    - final period pays coupon + principal
    """
    n = int(bond.maturity_years * bond.payment_frequency)
    dt = 1.0 / bond.payment_frequency
    times = [dt * (i + 1) for i in range(n)]
    cpn = bond.face_value * bond.coupon_rate / bond.payment_frequency
    cashflows = [cpn] * n
    cashflows[-1] += bond.face_value
    return times, cashflows


def price_bond_from_curve(
    bond: Bond,
    curve: DiscountCurve,
    clean: bool = True,
    settlement_time: float = 0.0,
) -> float:
    """
    Price bond using risk-free discount curve plus constant credit spread
    as a parallel shift in discount rates.
    The curve is assumed to provide the base term-structure; credit_spread is added.
    """
    times, cashflows = generate_cashflows(bond)
    df_list = []
    pv = 0.0
    accrued = accrued_interest(bond, settlement_time)

    for t, cf in zip(times, cashflows):
        effective_t = max(t - settlement_time, 0.0)
        base_df = curve.discount_factor(effective_t, comp="cont")
        # apply credit spread as extra discount
        df = base_df * math.exp(-bond.credit_spread * effective_t)
        df_list.append(df)
        pv += cf * df

    dirty_price = pv
    if clean:
        return dirty_price - accrued
    else:
        return dirty_price


def accrued_interest(bond: Bond, settlement_time: float) -> float:
    """
    Simple accrued interest under ACT/ACT-like assumption where year fraction
    since last coupon is settlement_time modulo coupon period.
    For a professional system, day count conventions should be explicit.
    """
    period = 1.0 / bond.payment_frequency
    if settlement_time <= 0:
        return 0.0
    # time since last coupon
    t_since_last = settlement_time % period
    year_frac = t_since_last / period
    coupon_per_period = bond.face_value * bond.coupon_rate / bond.payment_frequency
    return coupon_per_period * year_frac


def yield_to_maturity(
    bond: Bond,
    market_price: float,
    settlement_time: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Solve for yield to maturity (annualized, IRR) using Newton-Raphson on dirty price.
    """
    times, cashflows = generate_cashflows(bond)
    accrued = accrued_interest(bond, settlement_time)
    dirty_price_target = market_price + accrued

    # initial guess: coupon rate + credit spread approximation
    y = bond.coupon_rate + bond.credit_spread

    for _ in range(max_iter):
        pv = 0.0
        dpv = 0.0
        for t, cf in zip(times, cashflows):
            effective_t = max(t - settlement_time, 0.0)
            df = (1.0 + y / bond.payment_frequency) ** (-bond.payment_frequency * effective_t)
            pv += cf * df
            # derivative w.r.t y
            dpv += cf * df * (-effective_t * bond.payment_frequency / (1.0 + y / bond.payment_frequency))

        diff = pv - dirty_price_target
        if abs(diff) < tol:
            return y
        y_new = y - diff / dpv
        # keep yield reasonable
        if y_new <= -0.9999:
            y_new = -0.9999
        y = y_new

    return y


def macaulay_duration_convexity(
    bond: Bond,
    ytm: float,
    settlement_time: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute Macaulay duration and convexity under yield-to-maturity discounting.
    """
    times, cashflows = generate_cashflows(bond)
    period = 1.0 / bond.payment_frequency
    pv_total = 0.0
    weighted_time_sum = 0.0
    convexity_sum = 0.0

    for t, cf in zip(times, cashflows):
        effective_t = max(t - settlement_time, 0.0)
        m = bond.payment_frequency
        df = (1.0 + ytm / m) ** (-m * effective_t)
        pv_cf = cf * df
        pv_total += pv_cf
        weighted_time_sum += effective_t * pv_cf
        convexity_sum += effective_t * (effective_t + 1.0 / m) * pv_cf

    macaulay = weighted_time_sum / pv_total if pv_total != 0 else 0.0
    convexity = convexity_sum / (pv_total * (1.0 + ytm / bond.payment_frequency) ** 2) if pv_total != 0 else 0.0
    return macaulay, convexity


def modified_duration(macaulay_duration: float, ytm: float, freq: int) -> float:
    """Modified duration from Macaulay duration."""
    return macaulay_duration / (1.0 + ytm / freq)


def dv01(modified_duration_value: float, dirty_price: float) -> float:
    """
    DV01: price change in currency units for a 1 bp parallel shift in yield.
    DV01 ≈ modified_duration * dirty_price * 0.0001
    """
    return modified_duration_value * dirty_price * 0.0001


def z_spread(
    bond: Bond,
    curve: DiscountCurve,
    market_price: float,
    settlement_time: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Constant z-spread over the risk-free curve such that discounted cashflows equal
    observed dirty price. Z-spread is expressed in absolute terms (e.g. 0.015 = 150 bps). [web:4][web:5]
    """
    times, cashflows = generate_cashflows(bond)
    accrued = accrued_interest(bond, settlement_time)
    dirty_target = market_price + accrued

    z = bond.credit_spread  # initial guess

    for _ in range(max_iter):
        pv = 0.0
        dpv = 0.0
        for t, cf in zip(times, cashflows):
            effective_t = max(t - settlement_time, 0.0)
            base_df = curve.discount_factor(effective_t, comp="cont")
            df = base_df * math.exp(-z * effective_t)
            pv += cf * df
            dpv += cf * df * (-effective_t)
        diff = pv - dirty_target
        if abs(diff) < tol:
            return z
        z_new = z - diff / dpv
        z = z_new

    return z


def greenium(
    green_bond_z: float,
    comparable_vanilla_z: float,
) -> float:
    """
    Greenium defined as difference in z-spread between green bond and comparable
    vanilla bond (vanilla_z - green_z). Positive value => green bond trades tighter. [web:5][web:17]
    """
    return comparable_vanilla_z - green_bond_z


###############################################################################
# 3. ESG Scoring Engine
###############################################################################


@dataclass
class ESGInputs:
    # Environmental
    co2_avoided_tons_per_year: float          # higher is better
    energy_eff_improvement_pct: float         # higher is better
    renewable_share_pct: float                # higher is better
    water_reduction_pct: float                # higher is better
    climate_alignment_score: float            # already on 0–100

    # Social
    jobs_created: int                         # higher is better
    community_impact_score: float             # 0–100
    health_safety_score: float                # 0–100
    inclusion_access_score: float             # 0–100

    # Governance
    board_independence_pct: float            # 0–100
    esg_transparency_score: float            # 0–100
    reporting_frequency_per_year: int        # e.g. 1=annual, 4=quarterly
    external_audit: bool                     # True/False


@dataclass
class ESGScores:
    environmental_score: float
    social_score: float
    governance_score: float
    final_score: float
    rating: str
    climate_impact_indicators: Dict[str, float]


def _normalize_min_max(x: float, xmin: float, xmax: float) -> float:
    """Map x in [xmin, xmax] to [0, 100], clipping outside range."""
    if xmax <= xmin:
        return 0.0
    x_clipped = max(min(x, xmax), xmin)
    return 100.0 * (x_clipped - xmin) / (xmax - xmin)


def _governance_audit_score(external_audit: bool) -> float:
    """Simple binary scoring for external ESG audit/verification."""
    return 100.0 if external_audit else 40.0


def compute_esg_scores(inputs: ESGInputs) -> ESGScores:
    """
    Compute detailed ESG scores with:
    - Normalization of raw metrics to 0–100
    - Pillar weights: E=40%, S=30%, G=30% (common practice to ensure G weight is material). [web:6][web:12]
    - Within-pillar sub-factor weights are explicit and sum to 1.
    - Final 0–100 score and mapping to AAA–CCC based on breakpoints inspired by rating practices. [web:6][web:18]
    """

    # ----------------- Environmental sub-scores -----------------
    # Normalization ranges chosen to be realistic but conservative for project-level KPIs.
    e_co2 = _normalize_min_max(inputs.co2_avoided_tons_per_year, xmin=0.0, xmax=500_000.0)
    e_eff = _normalize_min_max(inputs.energy_eff_improvement_pct, xmin=0.0, xmax=50.0)
    e_ren = _normalize_min_max(inputs.renewable_share_pct, xmin=0.0, xmax=100.0)
    e_water = _normalize_min_max(inputs.water_reduction_pct, xmin=0.0, xmax=60.0)
    # climate_alignment_score is assumed already in 0–100 scale, but clipped
    e_climate = max(min(inputs.climate_alignment_score, 100.0), 0.0)

    # Environmental weights: decarbonization and climate alignment emphasized. [web:9][web:15]
    w_e = {
        "co2": 0.30,
        "eff": 0.20,
        "ren": 0.20,
        "water": 0.10,
        "climate": 0.20,
    }
    e_score = (
        w_e["co2"] * e_co2
        + w_e["eff"] * e_eff
        + w_e["ren"] * e_ren
        + w_e["water"] * e_water
        + w_e["climate"] * e_climate
    )

    # ----------------- Social sub-scores -----------------
    s_jobs = _normalize_min_max(inputs.jobs_created, xmin=0.0, xmax=10_000.0)
    s_comm = max(min(inputs.community_impact_score, 100.0), 0.0)
    s_hs = max(min(inputs.health_safety_score, 100.0), 0.0)
    s_inc = max(min(inputs.inclusion_access_score, 100.0), 0.0)

    # Social weights: local impact and inclusion prioritized. [web:9][web:15]
    w_s = {
        "jobs": 0.25,
        "comm": 0.25,
        "hs": 0.25,
        "inc": 0.25,
    }
    s_score = (
        w_s["jobs"] * s_jobs
        + w_s["comm"] * s_comm
        + w_s["hs"] * s_hs
        + w_s["inc"] * s_inc
    )

    # ----------------- Governance sub-scores -----------------
    g_board = _normalize_min_max(inputs.board_independence_pct, xmin=0.0, xmax=100.0)
    g_transp = max(min(inputs.esg_transparency_score, 100.0), 0.0)
    g_report_freq = _normalize_min_max(inputs.reporting_frequency_per_year, xmin=1.0, xmax=12.0)
    g_audit = _governance_audit_score(inputs.external_audit)

    # Governance weights: independence and transparency dominate; audit is a discrete enhancer. [web:6][web:12][web:18]
    w_g = {
        "board": 0.35,
        "transp": 0.30,
        "freq": 0.15,
        "audit": 0.20,
    }
    g_score = (
        w_g["board"] * g_board
        + w_g["transp"] * g_transp
        + w_g["freq"] * g_report_freq
        + w_g["audit"] * g_audit
    )

    # ----------------- Pillar aggregation -----------------
    # Overall ESG score weights: E=40%, S=30%, G=30%, broadly aligned with market weighting conventions. [web:6][web:12][web:15]
    final_score = 0.40 * e_score + 0.30 * s_score + 0.30 * g_score

    # ----------------- Rating mapping (AAA–CCC) -----------------
    # Breakpoints inspired by common ESG scales (e.g., MSCI AAA–CCC). [web:6][web:18]
    if final_score >= 85:
        rating = "AAA"
    elif final_score >= 77:
        rating = "AA"
    elif final_score >= 70:
        rating = "A"
    elif final_score >= 60:
        rating = "BBB"
    elif final_score >= 50:
        rating = "BB"
    elif final_score >= 40:
        rating = "B"
    else:
        rating = "CCC"

    climate_indicators = {
        "co2_avoided_tons_per_year": inputs.co2_avoided_tons_per_year,
        "energy_efficiency_improvement_pct": inputs.energy_eff_improvement_pct,
        "renewable_share_pct": inputs.renewable_share_pct,
        "water_reduction_pct": inputs.water_reduction_pct,
        "climate_alignment_score": e_climate,
    }

    return ESGScores(
        environmental_score=e_score,
        social_score=s_score,
        governance_score=g_score,
        final_score=final_score,
        rating=rating,
        climate_impact_indicators=climate_indicators,
    )


###############################################################################
# 4. Green Bond Regulatory / Classification Engine
###############################################################################


@dataclass
class GreenBondClassificationInputs:
    use_of_proceeds_aligned: bool          # clear environmental use-of-proceeds
    process_for_project_evaluation: bool   # documented selection process
    management_of_proceeds: bool           # tracked and ring-fenced
    reporting_commitment: bool             # regular allocation/impact reporting
    external_review: bool                  # second party opinion / verification

    eu_taxonomy_alignment_pct: float       # % of proceeds aligned with taxonomy criteria [web:19]
    primary_objective_sustainable: bool    # is sustainable investment the primary objective?
    promotes_esg_characteristics: bool     # does product promote E/S characteristics?


@dataclass
class GreenBondClassification:
    icma_compliant: bool
    icma_score: float
    eu_taxonomy_alignment_pct: float
    sfdr_article: str


def classify_green_bond(inputs: GreenBondClassificationInputs) -> GreenBondClassification:
    """
    Classify bond under:
    - ICMA Green Bond Principles (GBP)
    - EU Taxonomy alignment (simple % as provided)
    - SFDR Article 6 / 8 / 9 (applied in a bond/fund context) [web:7][web:10][web:16]
    """

    # ICMA GBP: four core components + external review. [web:16]
    icma_components = [
        inputs.use_of_proceeds_aligned,
        inputs.process_for_project_evaluation,
        inputs.management_of_proceeds,
        inputs.reporting_commitment,
        inputs.external_review,
    ]
    icma_score = 100.0 * sum(1 for c in icma_components if c) / len(icma_components)
    icma_compliant = icma_score >= 80.0  # simple threshold for “highly aligned”

    # EU Taxonomy: directly use the input percentage, but clip.
    eu_tax_pct = max(min(inputs.eu_taxonomy_alignment_pct, 100.0), 0.0)

    # SFDR classification logic (simplified, at product level). [web:7][web:10]
    # - Article 9: primary objective sustainable investment + high taxonomy alignment
    # - Article 8: promotes E/S characteristics but not pure sustainable objective
    # - Article 6: none of the above
    if inputs.primary_objective_sustainable and eu_tax_pct >= 50.0:
        sfdr = "Article 9"
    elif inputs.promotes_esg_characteristics:
        sfdr = "Article 8"
    else:
        sfdr = "Article 6"

    return GreenBondClassification(
        icma_compliant=icma_compliant,
        icma_score=icma_score,
        eu_taxonomy_alignment_pct=eu_tax_pct,
        sfdr_article=sfdr,
    )


###############################################################################
# 5. End-to-End KPI Dashboard
###############################################################################


def build_kpi_dashboard(
    green_bond: Bond,
    vanilla_bond: Bond,
    curve: DiscountCurve,
    esg_inputs: ESGInputs,
    gbp_inputs: GreenBondClassificationInputs,
    settlement_time: float = 0.0,
) -> Dict[str, Any]:
    """
    End-to-end KPI calculation:
    - Financials: price, YTM, durations, convexity, DV01, z-spread, credit spread, greenium.
    - ESG scores: E/S/G, final score, rating, climate indicators.
    - Regulatory: ICMA score/compliance, EU taxonomy %, SFDR article.
    """

    # ----------------- Financial KPIs: Green Bond -----------------
    green_clean_price = price_bond_from_curve(green_bond, curve, clean=True, settlement_time=settlement_time)
    green_dirty_price = price_bond_from_curve(green_bond, curve, clean=False, settlement_time=settlement_time)
    green_ytm = yield_to_maturity(green_bond, green_clean_price, settlement_time=settlement_time)
    green_mac_dur, green_convexity = macaulay_duration_convexity(green_bond, green_ytm, settlement_time)
    green_mod_dur = modified_duration(green_mac_dur, green_ytm, green_bond.payment_frequency)
    green_dv01 = dv01(green_mod_dur, green_dirty_price)
    green_z = z_spread(green_bond, curve, green_clean_price, settlement_time=settlement_time)
    green_credit_spread = green_bond.credit_spread

    # ----------------- Financial KPIs: Vanilla Bond -----------------
    vanilla_clean_price = price_bond_from_curve(vanilla_bond, curve, clean=True, settlement_time=settlement_time)
    vanilla_z = z_spread(vanilla_bond, curve, vanilla_clean_price, settlement_time=settlement_time)

    # Greenium
    greenium_value = greenium(green_z, vanilla_z)

    # ----------------- ESG KPIs -----------------
    esg_scores = compute_esg_scores(esg_inputs)

    # ----------------- Regulatory / Classification KPIs -----------------
    classification = classify_green_bond(gbp_inputs)

    # ----------------- Structured Dashboard -----------------
    dashboard = {
        "financial_kpis": {
            "clean_price": green_clean_price,
            "dirty_price": green_dirty_price,
            "ytm": green_ytm,
            "macaulay_duration_years": green_mac_dur,
            "modified_duration_years": green_mod_dur,
            "convexity": green_convexity,
            "dv01": green_dv01,
            "z_spread": green_z,
            "credit_spread_input": green_credit_spread,
            "greenium_vs_vanilla_z_spread": greenium_value,
            "vanilla_bond_z_spread": vanilla_z,
        },
        "esg_kpis": {
            "environmental_score": esg_scores.environmental_score,
            "social_score": esg_scores.social_score,
            "governance_score": esg_scores.governance_score,
            "final_esg_score": esg_scores.final_score,
            "esg_rating": esg_scores.rating,
            "climate_impact_indicators": esg_scores.climate_impact_indicators,
        },
        "regulatory_kpis": {
            "icma_compliance": classification.icma_compliant,
            "icma_score": classification.icma_score,
            "eu_taxonomy_alignment_pct": classification.eu_taxonomy_alignment_pct,
            "sfdr_article": classification.sfdr_article,
        },
    }
    return dashboard


###############################################################################
# 6. Example Usage (Deterministic, for Testing)
###############################################################################


def example_usage() -> None:
    """
    Example deterministic configuration so that the script yields a stable output.
    This can be adapted to real market data in production.
    """

    # -------- Discount curve (simple stylized euro risk-free curve in cont. comp.) [web:1][web:14] --------
    curve = DiscountCurve(
        times=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        zero_rates=[0.015, 0.017, 0.019, 0.020, 0.022, 0.023, 0.025],
    )

    # -------- Green bond definition (5y, semi-annual, 2% coupon) --------
    green_bond = Bond(
        face_value=1_000.0,
        coupon_rate=0.02,
        payment_frequency=2,
        maturity_years=5.0,
        issue_price=100.0,       # not used in pricing, but could be stored
        credit_spread=0.010,     # 100 bps
    )

    # -------- Comparable vanilla bond (same issuer, same maturity, slightly wider spread) --------
    vanilla_bond = Bond(
        face_value=1_000.0,
        coupon_rate=0.02,
        payment_frequency=2,
        maturity_years=5.0,
        issue_price=100.0,
        credit_spread=0.0115,    # 115 bps, reflecting a small greenium. [web:5][web:11]
    )

    # -------- ESG quantitative inputs (illustrative but realistic project) --------
    esg_inputs = ESGInputs(
        co2_avoided_tons_per_year=150_000.0,
        energy_eff_improvement_pct=30.0,
        renewable_share_pct=85.0,
        water_reduction_pct=25.0,
        climate_alignment_score=80.0,
        jobs_created=1_200,
        community_impact_score=75.0,
        health_safety_score=88.0,
        inclusion_access_score=82.0,
        board_independence_pct=65.0,
        esg_transparency_score=90.0,
        reporting_frequency_per_year=4,
        external_audit=True,
    )

    # -------- Green Bond Principles & regulatory inputs --------
    gbp_inputs = GreenBondClassificationInputs(
        use_of_proceeds_aligned=True,
        process_for_project_evaluation=True,
        management_of_proceeds=True,
        reporting_commitment=True,
        external_review=True,
        eu_taxonomy_alignment_pct=65.0,  # majority aligned with EU taxonomy criteria. [web:19]
        primary_objective_sustainable=True,
        promotes_esg_characteristics=True,
    )

    dashboard = build_kpi_dashboard(
        green_bond=green_bond,
        vanilla_bond=vanilla_bond,
        curve=curve,
        esg_inputs=esg_inputs,
        gbp_inputs=gbp_inputs,
        settlement_time=0.0,
    )

    # Pretty-print as a structured dictionary
    import pprint
    pprint.pprint(dashboard)


if __name__ == "__main__":
    # Running the module directly executes the example and prints the KPI dashboard.
    example_usage()
