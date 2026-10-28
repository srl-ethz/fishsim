# Table inventory

Every table produced for this work, where it currently lives, and the LaTeX for the ones
that are not in the paper. Generated from the report sources and the two paper revisions.

| # | table | paper §  | in report | in the Aug draft | in current paper | report chapter |
|---|---|---|---|---|---|---|
| 1 | Per-joint tail stiffness and damping (`tab:stiff`) | IV.A | yes | yes | cut | report ch. 4 |
| 2 | Identified quasi-steady fluid coefficients (E13a) (`tab:fluid`) | IV.B | yes | yes | in paper | report ch. 4 |
| 3 | Motor actuator parameters: inherited vs identified (`tab:motor`) | IV.C | yes | yes | in paper | report ch. 4 |
| 4 | Observation-noise model injected during training (`tab:sensornoise`) | V.C | yes | yes | cut | report ch. 4 |
| 5 | Vision frame rate: ROI tracking vs full-frame (`tab:framerate`) | VI.A | yes | — | cut | report ch. 4 |
| 6 | The deployed policy's 24-D observation / 6-D action (`tab:obsact`) | V.C | yes | yes | cut | report ch. 4 |
| 7 | Training curriculum (ten stages, three phases) (`tab:curriculum`) | V.C | — | yes | cut | not in report |
| 8 | Spectral content by policy and gait direction (`tab:power`) | VI.C | yes | yes | in paper | report ch. 5 |
| 9 | Deployment reach, 2026-05-20 session (`tab:deploy-reach`) | VI.B | yes | yes | cut | report ch. 5 |
| 10 | CPG scaling constants, training vs deployment (`tab:scaling`) | VI.B | yes | yes | in paper | report ch. 5 |
| 11 | Direct-action twin fluid coefficients (refitB) (`tab:fluidsets`) | appendix | yes | — | cut | report appendix |

The paper carries 4 of the 11. `tab:fluid` and `tab:motor` were restored on 9 September.
Not in the paper: `tab:stiff`, `tab:sensornoise`, `tab:framerate`, `tab:obsact`,
`tab:curriculum`, `tab:deploy-reach`, `tab:fluidsets`.

---

## LaTeX

Bodies below are taken from the report sources. The paper versions were already reformatted
to IEEE two-column width; recover those with:

```bash
git -C 06_Publication/paper show 420c794:Chapters/04SystemIdentification.tex
```

### 1. Per-joint tail stiffness and damping — `tab:stiff`

*Per-joint tail stiffness and damping, from beam theory (\Cref{eq:beam}) calibrated to the measured shaft stiffness $k_\mathrm{eff}\approx 0.98$~N$\cdot$m/rad. Joint~1 is at the body, joint~5 at the caudal fin; stiffness decreases toward the tail.*

```latex
\begin{tabular}{lccccc}
\toprule
Joint (body $\to$ fin)          & 1 & 2 & 3 & 4 & 5 \\
\midrule
Stiffness [N\,m/rad]            & $2.99$   & $2.39$   & $2.12$   & $1.77$   & $1.32$ \\
Damping [N\,m\,s/rad]           & $0.033$  & $0.027$  & $0.024$  & $0.020$  & $0.015$ \\
\bottomrule
\end{tabular}
```

### 2. Identified quasi-steady fluid coefficients (E13a) — `tab:fluid`

*Identified quasi-steady fluid coefficients (model E13a), five per segment. The fin's angular drag and Kutta lift are markedly larger than the head and tail values, the stateless model inflating them to stand in for unsteady thrust it cannot represent.*

```latex
\begin{tabular}{lccccc}
\toprule
Segment & blunt drag & slender drag & angular drag & Kutta lift & Magnus lift \\
\midrule
Head & $0.006$ & $0.045$ & $0.813$  & $0.521$  & $0.903$ \\
Tail & $0.039$ & $0.551$ & $5.428$  & $0.001$  & $1.581$ \\
Fin  & $0.171$ & $0.004$ & $18.310$ & $10.148$ & $5.951$ \\
\bottomrule
\end{tabular}
```

### 3. Motor actuator parameters: inherited vs identified — `tab:motor`

*Motor actuator parameters: the inherited ``ideal'' model versus the identified one. The position gain and rotor armature are fitted to the deployment tracking data; the force limit is the hardware value; the joint damping is set to cap the peak speed near $5$~*

```latex
\begin{tabular}{lcc}
\toprule
Parameter & Old (ideal) & Identified \\
\midrule
Position gain $k_p$                 & $800$   & $5.41$ \\
Velocity gain $k_v$                 & $20$    & $0$ \\
Rotor armature [kg\,m$^2$]          & $0.001$ & $0.0113$ \\
Joint damping $b$ [N\,m\,s/rad]     & $0$     & $0.48$ (set) \\
Force limit [N$\cdot$m]             & $1.0$   & $2.3$ \\
Transport delay $\tau_d$ [ms]       & $0$     & $\approx\!14$ \\
\bottomrule
\end{tabular}
```

### 4. Observation-noise model injected during training — `tab:sensornoise`

*Observation-noise model injected during training (the \texttt{SensorModel}). Each channel's training value is set at or above the value measured on the SOFI hardware. ``set'' marks the few terms fixed by engineering margin rather than a dedicated measurement; *

```latex
\begin{tabular}{@{}llll@{}}
\toprule
Observation & Training value & Hardware basis & Identified from \\
\midrule
\multicolumn{4}{@{}l}{\emph{Proprioception (per control step, $50$~Hz)}}\\
Encoder position           & $\sigma=0.001$~rad     & $0.6$--$2.3\!\times\!10^{-3}$~rad & stationary-encoder std \\
Encoder velocity           & $\sigma=0.01$~rad/s    & --- (set)                        & engineering margin \\
IMU gyro $\omega_z$ (white) & $\sigma=0.006$~rad/s   & $0.002$~rad/s                    & stationary-gyro std \\
IMU gyro bias (per episode) & $\sigma=0.005$~rad/s   & ${\sim}5\!\times\!10^{-5}$~rad/s  & per-run offset (margin) \\
\midrule
\multicolumn{4}{@{}l}{\emph{Vision\,/\,AprilTag (per delivered frame)}}\\
Tag position $(x,y)$       & $\sigma=0.002$~m       & $0.0013$~m                       & static-tag scatter \\
Tag velocity               & $\sigma=0.02$~m/s      & $0.016$~m/s                      & finite diff.\ of static tag \\
Tag heading                & $\sigma=1.5^\circ$     & --- (set)                        & angular margin \\
Pipeline latency           & $U(10,50)$~ms          & $p_{50}\,25$, $p_{95}\,51$~ms    & recorded frame age \\
Frame-drop probability     & $0.10$                 & $0.15$--$0.39$                   & frame-sequence gaps \\
Camera rate (CPG)          & $30$~Hz, fixed         & mean $53.2$~Hz                   & inter-frame intervals \\
\quad FPS\,+\,jitter DR (DA only) & $U(39.4,66.9)$~Hz, $12$~ms & $12.5$~ms                   & inter-frame std \\
\bottomrule
\end{tabular}
```

### 5. Vision frame rate: ROI tracking vs full-frame — `tab:framerate`

*Vision frame rate: region-of-interest tracking versus full-frame scanning, at two detector decimation factors. ROI tracking keeps detection above the $50$~Hz control rate; decimation trades field of view for speed.*

```latex
\begin{tabular}{lcc}
\toprule
 & Decimate $2$ & Decimate $4$ \\
\midrule
Full frame & ${\sim}30$~fps    & ${\sim}40$~fps \\
ROI        & $35$--$40$~fps    & ${\sim}60$~fps \\
\bottomrule
\end{tabular}
```

### 6. The deployed policy's 24-D observation / 6-D action — `tab:obsact`

*The deployed policy's $24$-dimensional observation and $6$-dimensional action.*

```latex
\begin{tabular}{llc}
\toprule
Group & Contents & Dim. \\
\midrule
\multicolumn{3}{l}{\emph{Observation (24)}}\\
\quad Vision (target)      & relative target pose $+$ 12-step history & $12$ \\
\quad Proprioception       & motor encoder position ($\cos,\sin$), speed & $3$ \\
\quad IMU                  & body yaw rate & $1$ \\
\quad Controller state     & CPG phase ($\cos,\sin$) $+$ action history & $8$ \\
\midrule
\multicolumn{3}{l}{\emph{Action (6, CPG parameters)}}\\
\quad Amplitudes/phases    & $k_{1c},k_{1s},k_{2c},k_{2s}$ & $4$ \\
\quad Frequency            & $\omega$ & $1$ \\
\quad Turn bias            & $\theta_\mathrm{center}$ & $1$ \\
\bottomrule
\end{tabular}
```

### 7. Training curriculum (ten stages, three phases) — `tab:curriculum`

Written for the paper, not in the report. Recover from the August paper revision:
`git -C 06_Publication/paper show 420c794:Chapters/05Control.tex`.

### 8. Spectral content by policy and gait direction — `tab:power`

*Spectral content by policy and gait direction under domain randomisation ($n=30$ episodes each). ``amp'' is the actual motor amplitude; ``cmd${>}4$'' and ``vel${>}4$/${>}8$'' are the fractions of commanded-setpoint and actual motor-\emph{velocity} power above *

```latex
\begin{tabular}{llccccc}
\toprule
Policy & gait & $n$ & amp (rad) & cmd${>}4$ & vel${>}4$ & vel${>}8$ \\
\midrule
CPG     & forward  & 25 & 1.43 & 7\%  & 4\%  & ${<}1\%$ \\
        & backward &  3 & 1.55 & 6\%  & 6\%  & 1\%  \\
DA      & forward  & 19 & 0.79 & 67\% & 50\% & 28\% \\
        & backward &  8 & 0.94 & 67\% & 67\% & 38\% \\
DA-LPF1 & forward  & 18 & 1.48 & 3\%  & 6\%  & 2\%  \\
        & backward &  9 & 0.84 & 22\% & 55\% & 12\% \\
\bottomrule
\end{tabular}
```

### 9. Deployment reach, 2026-05-20 session — `tab:deploy-reach`

*Deployment reach, 2026-05-20 session: the same $30$ runs grouped three ways, each summing to $20/30$ overall. Bearing bands are lateral ($60^\circ$--$90^\circ$), lateral-rear ($90^\circ$--$120^\circ$, the hardest), and behind ($>120^\circ$, out-of-distribution*

```latex
\begin{tabular}{lc@{\hspace{2.2em}}lc@{\hspace{2.2em}}lc}
\toprule
\multicolumn{2}{c}{By type} & \multicolumn{2}{c}{By target bearing $|\theta|$} & \multicolumn{2}{c}{By side} \\
\cmidrule(r){1-2}\cmidrule(lr){3-4}\cmidrule(l){5-6}
Static      & $13/22$ & $\le 60^\circ$           & $14/16$ & right ($\theta<0$)    & $10/12$ \\
Disturbance & $7/8$   & $60^\circ$--$90^\circ$    & $4/8$   & left ($\theta\ge 0$)  & $10/18$ \\
            &         & $90^\circ$--$120^\circ$   & $0/4$   &                       &         \\
            &         & $>120^\circ$              & $2/2$   &                       &         \\
\midrule
\multicolumn{6}{c}{Overall: $20/30$ ($67\%$)}\\
\bottomrule
\end{tabular}
```

### 10. CPG scaling constants, training vs deployment — `tab:scaling`

*CPG scaling constants at training versus deployment. The two sets drifted apart across repositories and were not re-synchronised before the session. Amplitudes and the offset are in radians; the frequency entry is $\omega_\mathrm{max}$, which both scales and c*

```latex
\begin{tabular}{lccl}
\toprule
Constant & Training & Deployment & Change \\
\midrule
Amplitude $K_1^{\max}$        & $0.80$ & $0.30$ & $2.7\times$ smaller \\
2nd-harmonic $K_2^{\max}$     & $0.40$ & $0.15$ & $2.7\times$ smaller \\
Turn bias $\theta_\mathrm{center}$ & $0.80$ & $0.15$ & $5.3\times$ smaller \\
Frequency ceiling             & $4$~Hz & $6$~Hz & $1.5\times$ higher \\
Output clamp                  & $0.80$~rad & $0.65$~rad & tighter \\
\bottomrule
\end{tabular}
```

### 11. Direct-action twin fluid coefficients (refitB) — `tab:fluidsets`

*Direct-action twin fluid coefficients (refitB), five per segment. The CPG twin's E13a set is in \Cref{tab:fluid}.*

```latex
\begin{tabular}{lccccc}
\toprule
Segment & blunt drag & slender drag & angular drag & Kutta lift & Magnus lift \\
\midrule
Head & $0.003$ & $0.007$ & $3.913$  & $0.647$  & $1.996$ \\
Tail & $0.064$ & $2.350$ & $12.356$ & $0.469$  & $5.145$ \\
Fin  & $0.146$ & $1.581$ & $2.002$  & $14.307$ & $5.411$ \\
\bottomrule
\end{tabular}
```

