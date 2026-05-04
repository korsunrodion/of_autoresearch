'use strict';
require('dotenv').config({ path: require('path').join(__dirname, '../.env') });
const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');
const { predict, FEATURES, THRESHOLDS } = require('./classifier.js');

// ── Constants (must match Python) ─────────────────────────────────────────────
const WINDOWS_MS = [30,60,240,360,420,480,600,720,1440,10080,20160,40320].map(m => m * 60 * 1000);
const WINDOW_STRS = ['0h30m','1h','4h','6h','7h','8h','10h','12h','24h','168h','336h','672h'];
const ID_RANGES  = [10000,20000,50000,100000,200000,500000,1000000];
const RANGE_TAGS = ['10k','20k','50k','100k','200k','500k','1m'];
const KEY_WS     = new Set(['6h','7h','8h','10h','12h','24h']);
const SHORT_WS   = new Set(['0h30m','1h','4h','6h','7h','8h']);
const DENSITY_WH = [1,4,6,12,24];
const DENSITY_MS = DENSITY_WH.map(h => h * 3600 * 1000);
const MAX_W_MS   = WINDOWS_MS[WINDOWS_MS.length - 1];
const _5MIN_MS   = 5 * 60 * 1000;

const RISK_MAP = { 'No risk':1, 'Low':2, 'High':3, 'Very High':4, 'Extreme':5 };

// ── Username helpers ───────────────────────────────────────────────────────────
function isUDigits(name) {
  return (typeof name === 'string' && name.length > 1 && name[0] === 'u' && /^\d+$/.test(name.slice(1))) ? 1.0 : 0.0;
}
function digitRatio(name) {
  if (typeof name !== 'string' || name.length === 0) return 0.0;
  return name.split('').filter(c => c >= '0' && c <= '9').length / name.length;
}
function entropy(name) {
  if (typeof name !== 'string' || name.length === 0) return 0.0;
  const counts = {};
  for (const c of name) counts[c] = (counts[c] || 0) + 1;
  const n = name.length;
  return -Object.values(counts).reduce((s, c) => s + (c/n) * Math.log2(c/n), 0);
}

// ── Pairwise feature computation for one model group ─────────────────────────
function computeModelFeatures(rows, crossModel, noriskMed, noriskRate, riskFracs) {
  const n = rows.length;

  // Build pairwise time/id diff arrays (flat n×n)
  const tdiff  = new Float64Array(n * n);
  const iddiff = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      tdiff[i*n+j]  = Math.abs(rows[i].t - rows[j].t);
      iddiff[i*n+j] = Math.abs(rows[i].id - rows[j].id);
    }
  }

  // Sorted arrays for percentile computation
  const sortedIds = rows.map(r => r.id).sort((a,b) => a-b);
  const sortedTs  = rows.map(r => r.t).sort((a,b) => a-b);
  const modelFirstT = sortedTs[0];

  const result = [];

  for (let i = 0; i < n; i++) {
    const row = {};
    let nEmptyShort = 0;

    // ── Window features ────────────────────────────────────────────────────
    for (let wi = 0; wi < WINDOWS_MS.length; wi++) {
      const wMs  = WINDOWS_MS[wi];
      const wStr = WINDOW_STRS[wi];
      const isKey   = KEY_WS.has(wStr);
      const isShort = SHORT_WS.has(wStr);

      // Collect partners: j where |t_i - t_j| <= wMs and j != i
      const diffs = [], pids = [];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        if (tdiff[i*n+j] <= wMs) {
          diffs.push(iddiff[i*n+j]);
          pids.push(rows[j].id);
        }
      }

      if (diffs.length === 0) {
        row[`log_min_id_diff_${wStr}`] = NaN;
        row[`partner_count_${wStr}`]   = NaN;
        for (const tag of RANGE_TAGS) row[`cic_${tag}_${wStr}`] = NaN;
        if (isKey) {
          for (const tag of RANGE_TAGS) row[`cic_frac_${tag}_${wStr}`] = NaN;
          row[`log_id_span_${wStr}`]        = NaN;
          row[`log_id_std_${wStr}`]         = NaN;
          row[`log_min_consec_gap_${wStr}`] = NaN;
        }
        if (isShort) nEmptyShort++;
      } else {
        const pc    = diffs.length;
        const minD  = Math.min(...diffs);
        row[`log_min_id_diff_${wStr}`] = Math.log1p(minD);
        row[`partner_count_${wStr}`]   = pc;
        for (let ti = 0; ti < RANGE_TAGS.length; ti++) {
          const cic = diffs.filter(d => d <= ID_RANGES[ti]).length;
          row[`cic_${RANGE_TAGS[ti]}_${wStr}`] = cic;
          if (isKey) row[`cic_frac_${RANGE_TAGS[ti]}_${wStr}`] = cic / (pc + 1);
        }
        if (isKey) {
          const minPid = Math.min(...pids), maxPid = Math.max(...pids);
          row[`log_id_span_${wStr}`] = Math.log1p(maxPid - minPid);
          const mean = pids.reduce((s,v) => s+v, 0) / pc;
          const std  = pc > 1 ? Math.sqrt(pids.reduce((s,v) => s+(v-mean)**2, 0) / pc) : 0;
          row[`log_id_std_${wStr}`] = Math.log1p(std);
          if (pc >= 2) {
            const sorted = [...pids].sort((a,b) => a-b);
            const minGap = Math.min(...sorted.slice(1).map((v,k) => v - sorted[k]));
            row[`log_min_consec_gap_${wStr}`] = Math.log1p(minGap);
          } else {
            row[`log_min_consec_gap_${wStr}`] = NaN;
          }
        }
      }
    }

    // Forward-fill empty narrow windows from next wider window
    for (let j = 0; j < WINDOWS_MS.length - 1; j++) {
      const ws = WINDOW_STRS[j], nws = WINDOW_STRS[j+1];
      for (const base of ['log_min_id_diff','partner_count',...RANGE_TAGS.map(t=>`cic_${t}`)]) {
        const k = `${base}_${ws}`, nk = `${base}_${nws}`;
        if (isNaN(row[k])) row[k] = row[nk];
      }
    }

    // ── Density features ──────────────────────────────────────────────────
    for (let di = 0; di < DENSITY_WH.length; di++) {
      const wH  = DENSITY_WH[di];
      const dMs = DENSITY_MS[di];
      let cnt = 0;
      for (let j = 0; j < n; j++) if (tdiff[i*n+j] <= dMs) cnt++;
      const wDays = wH * 2 / 24;
      row[`model_subs_${wH}h`]      = cnt;
      row[`model_subs_rate_${wH}h`] = cnt / Math.max(noriskRate * wDays, 1.0);
    }

    // ── Global ID features ────────────────────────────────────────────────
    const gDiffs = [], gPids = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      gDiffs.push(iddiff[i*n+j]);
      gPids.push(rows[j].id);
    }
    if (gDiffs.length > 0) {
      row['log_min_id_diff_global'] = Math.log1p(Math.min(...gDiffs));
      row['log_global_id_span']     = Math.log1p(Math.max(...gPids) - Math.min(...gPids));
    } else {
      row['log_min_id_diff_global'] = NaN;
      row['log_global_id_span']     = NaN;
    }

    // ── Minimum time diff ─────────────────────────────────────────────────
    const tDiffsInWindow = [];
    for (let j = 0; j < n; j++) {
      if (j !== i && tdiff[i*n+j] <= MAX_W_MS) tDiffsInWindow.push(tdiff[i*n+j]);
    }
    row['log_min_time_diff'] = tDiffsInWindow.length > 0
      ? Math.log1p(Math.min(...tDiffsInWindow) / 60000) : NaN;

    // ── Norisk median features ────────────────────────────────────────────
    row['log_model_norisk_median_id_diff'] = isFinite(noriskMed) ? Math.log1p(noriskMed) : NaN;
    const raw24h = isNaN(row['log_min_id_diff_24h']) ? NaN : Math.expm1(row['log_min_id_diff_24h']);
    row['rel_min_id_diff_24h'] = (!isNaN(raw24h) && isFinite(noriskMed) && noriskMed > 0)
      ? raw24h / noriskMed : NaN;

    // ── Cross-model user features ─────────────────────────────────────────
    const cm = crossModel[rows[i].user_name] || { count: 1, maxRisk: rows[i].riskScore };
    row['user_model_count']        = cm.count;
    row['user_max_risk_elsewhere'] = cm.count > 1 ? cm.maxRisk : 0;
    row['log_user_id_num']         = isFinite(rows[i].id) ? Math.log1p(rows[i].id) : NaN;

    // ── Ratio features ────────────────────────────────────────────────────
    const c6    = row['cic_10k_6h'],  c12   = row['cic_10k_12h'];
    const c50_6 = row['cic_50k_6h'],  c50_24 = row['cic_50k_24h'];
    const pc6   = row['partner_count_6h'], pc24 = row['partner_count_24h'];
    row['cic_ratio_6h_24h']    = !isNaN(c50_6) ? c50_6 / (c50_24 + 1) : NaN;
    row['cic10k_ratio_6h_12h'] = !isNaN(c6)    ? c6    / (c12    + 1) : NaN;
    row['pc_ratio_6h_24h']     = !isNaN(pc6)   ? pc6   / (pc24   + 1) : NaN;

    // ── Time / username / model scalars ───────────────────────────────────
    const hour = (rows[i].ts % 86400) / 3600;
    row['hour_sin']    = Math.sin(2 * Math.PI * hour / 24);
    row['hour_cos']    = Math.cos(2 * Math.PI * hour / 24);
    row['day_of_week'] = ((Math.floor(rows[i].ts / 86400) + 4) % 7);
    row['total_chargebacks']        = rows[i].chargebacks;
    row['frac_empty_short_windows'] = nEmptyShort / SHORT_WS.size;
    row['username_is_u_digits']  = isUDigits(rows[i].user_name);
    row['username_len']          = typeof rows[i].user_name === 'string' ? rows[i].user_name.length : NaN;
    row['model_norisk_rate']     = noriskRate;
    row['username_digit_ratio']  = digitRatio(rows[i].user_name);
    row['username_entropy']      = entropy(rows[i].user_name);
    row['model_vh_frac']         = riskFracs.vh;
    row['model_extreme_frac']    = riskFracs.extreme;
    row['model_any_risk_frac']   = riskFracs.any;

    // ── Percentile / timing features ──────────────────────────────────────
    const idBisect = sortedIds.filter(v => v <= rows[i].id).length;
    const tBisect  = sortedTs.filter(v  => v <= rows[i].t).length;
    row['id_percentile_in_model']         = idBisect / n;
    row['time_percentile_in_model']       = tBisect  / n;
    row['log_time_since_model_first_sub'] = Math.log1p((rows[i].t - modelFirstT) / 3600000);
    let cnt5m = 0;
    for (let j = 0; j < n; j++) if (tdiff[i*n+j] <= _5MIN_MS) cnt5m++;
    row['n_subs_at_sub_time_5m'] = cnt5m;

    result.push({ user_name: rows[i].user_name, risk_level: rows[i].riskLevel, features: row });
  }

  return result;
}

// ── Norisk median per model ───────────────────────────────────────────────────
function computeNoriskMedian(rows) {
  const norisk = rows.filter(r => r.riskLevel === 'No risk');
  if (norisk.length === 0) return NaN;

  const minDiffs = [];
  for (const nr of norisk) {
    let minD = Infinity;
    for (const r of rows) {
      if (r.user_name === nr.user_name) continue;
      if (Math.abs(nr.t - r.t) <= MAX_W_MS) {
        const d = Math.abs(nr.id - r.id);
        if (d < minD) minD = d;
      }
    }
    if (isFinite(minD)) minDiffs.push(minD);
  }
  if (minDiffs.length === 0) return NaN;
  minDiffs.sort((a,b) => a-b);
  const mid = Math.floor(minDiffs.length / 2);
  return minDiffs.length % 2 === 1 ? minDiffs[mid] : (minDiffs[mid-1] + minDiffs[mid]) / 2;
}

// ── Norisk rate per model ─────────────────────────────────────────────────────
function computeNoriskRate(rows) {
  const nr = rows.filter(r => r.riskLevel === 'No risk');
  if (nr.length < 2) return 1.0;
  const ts = nr.map(r => r.t).sort((a,b) => a-b);
  const rangeDays = (ts[ts.length-1] - ts[0]) / 86400000;
  return nr.length / Math.max(rangeDays, 1.0);
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  const args = process.argv.slice(2);
  const outputFlag = args.indexOf('--output');
  const outputFile = outputFlag >= 0 ? args[outputFlag+1] : null;
  const usersFlag  = args.indexOf('--users');
  const filterUsers = usersFlag >= 0
    ? args.slice(usersFlag+1).filter(a => !a.startsWith('--'))
    : null;

  const pool = new Pool({ connectionString: process.env.DATABASE_URL });
  console.log('Fetching data from DB...');

  const { rows: dbRows } = await pool.query(`
    SELECT user_id, user_name, subscribed_at, total_chargebacks,
           risk_level, tracking_model_name
    FROM subscriptions
    WHERE risk_level IS NOT NULL
      AND user_name IS NOT NULL
      AND subscribed_at IS NOT NULL
  `);
  await pool.end();
  console.log(`  ${dbRows.length} rows`);

  // Clean and parse
  const data = [];
  for (const r of dbRows) {
    const riskScore = RISK_MAP[r.risk_level];
    if (!riskScore) continue;
    const t  = new Date(r.subscribed_at).getTime();
    const id = parseFloat(r.user_id);
    if (isNaN(t) || isNaN(id) || !r.user_name || !r.tracking_model_name) continue;
    data.push({
      user_name: r.user_name,
      tracking_model_name: r.tracking_model_name,
      t,                              // ms timestamp
      ts: t / 1000,                   // unix seconds
      id,
      riskLevel: r.risk_level,
      riskScore,
      chargebacks: parseFloat(r.total_chargebacks) || 0,
    });
  }
  console.log(`  ${data.length} valid rows after cleaning`);

  // Cross-model user stats
  console.log('Computing cross-model stats...');
  const crossModel = {};
  for (const r of data) {
    if (!crossModel[r.user_name]) crossModel[r.user_name] = { models: new Set(), maxRisk: 0 };
    crossModel[r.user_name].models.add(r.tracking_model_name);
    if (r.riskScore > crossModel[r.user_name].maxRisk)
      crossModel[r.user_name].maxRisk = r.riskScore;
  }
  for (const u of Object.keys(crossModel)) {
    crossModel[u] = { count: crossModel[u].models.size, maxRisk: crossModel[u].maxRisk };
  }

  // Group by model
  const byModel = {};
  for (const r of data) {
    if (!byModel[r.tracking_model_name]) byModel[r.tracking_model_name] = [];
    byModel[r.tracking_model_name].push(r);
  }
  const modelNames = Object.keys(byModel);
  console.log(`  ${modelNames.length} model groups`);

  // Compute features and predict per model group
  console.log('Computing features and predicting...');
  const t0 = Date.now();
  const allResults = [];
  let done = 0;

  for (const modelName of modelNames) {
    const rows = byModel[modelName];
    const n    = rows.length;

    // Skip models not containing filtered users
    if (filterUsers && !rows.some(r => filterUsers.includes(r.user_name))) continue;

    const noriskMed  = computeNoriskMedian(rows);
    const noriskRate = computeNoriskRate(rows);
    const nTotal     = rows.length;
    const riskFracs  = {
      vh:      rows.filter(r => r.riskLevel === 'Very High').length / nTotal,
      extreme: rows.filter(r => r.riskLevel === 'Extreme').length   / nTotal,
      any:     rows.filter(r => r.riskLevel !== 'No risk').length   / nTotal,
    };

    const userFeatures = computeModelFeatures(rows, crossModel, noriskMed, noriskRate, riskFracs);

    for (const { user_name, risk_level, features } of userFeatures) {
      if (filterUsers && !filterUsers.includes(user_name)) continue;

      // Build feature vector in FEATURES order (NaN for missing)
      const vec = FEATURES.map(f => {
        const v = features[f];
        return (v === undefined || v === null) ? NaN : v;
      });

      const pred = predict(vec);
      allResults.push({
        user_name,
        true_risk: risk_level,
        predicted_risk: pred.risk,
        vh_proba:      pred.vh.toFixed(4),
        extreme_proba: pred.extreme.toFixed(4),
        high_proba:    pred.high.toFixed(4),
        low_proba:     pred.low.toFixed(4),
      });
    }

    done++;
    if (done % 100 === 0) process.stdout.write(`\r  ${done}/${modelNames.length} models...`);
  }
  console.log(`\r  ${done}/${modelNames.length} models — done in ${((Date.now()-t0)/1000).toFixed(1)}s`);

  // Distribution
  const dist = {};
  for (const r of allResults) dist[r.predicted_risk] = (dist[r.predicted_risk] || 0) + 1;
  console.log('\nDistribution:');
  for (const [k,v] of Object.entries(dist).sort((a,b)=>b[1]-a[1]))
    console.log(`  ${k.padEnd(12)}: ${v}`);

  // Output
  if (outputFile) {
    const header = 'user_name,true_risk,predicted_risk,vh_proba,extreme_proba,high_proba,low_proba';
    const lines  = allResults.map(r =>
      `${r.user_name},${r.true_risk},${r.predicted_risk},${r.vh_proba},${r.extreme_proba},${r.high_proba},${r.low_proba}`);
    fs.writeFileSync(outputFile, [header, ...lines].join('\n') + '\n');
    console.log(`\nSaved ${allResults.length} predictions to ${outputFile}`);
  }
}

main().catch(err => { console.error(err); process.exit(1); });
