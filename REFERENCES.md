# References

All entries verified against OpenAlex on 2026-10-30 (OpenAlex work id in brackets).

## Clinical context and workflow

- Vigil C, Lasso A, Ghosh RM, Pinter C, Cianciulli A, Nam HH, Abid A, Herz C, Mascio CE, Chen J, Fuller S, Whitehead KK, Jolley MA. Modeling tool for rapid virtual planning of the intracardiac baffle in double-outlet right ventricle. *Ann Thorac Surg* 2021;111(6):2078-2083. doi:10.1016/j.athoracsur.2021.02.058 [W3135075326]
  Baffle built by thin-plate-spline warp of a flat disk to the boundary curve plus surface points; flattened with an adapted LSCM.
- Kizilski SB, Recco DP, Davee JM, Masterson A, Yao J, Earley PD, Kneier NE, Aycock KI, Craven BA, Pathmanathan P, Hammer PE, Hoganson DM. Preclinical validation of a patient-specific patch-planning workflow for congenital cardiovascular reconstruction. *Ann Biomed Eng* 2025;53(12):3505-3522. doi:10.1007/s10439-025-03870-4 [W4415163065]
  FE flattening (FEFlatten), 0.1 mm / 0.5 % tolerance on patch width and length, darts for elliptical or saddle-shaped patches, 1-1.25 mm suture-bite offset.
- Gobbi DG, Peters TM. Generalized 3D nonlinear transformations for medical imaging: an object-oriented implementation in VTK. *Comput Med Imaging Graph* 2003;27(4):255-265. doi:10.1016/s0895-6111(02)00091-5 [W2031408555]
  `vtkThinPlateSplineTransform`, used by the Baffle Planner and by the synthetic generator here.
- Bookstein FL. Principal warps: thin-plate splines and the decomposition of deformations. *IEEE Trans Pattern Anal Mach Intell* 1989;11(6):567-585. doi:10.1109/34.24792 [W2128409098]

## Flattening / parameterisation methods

- Tutte WT. How to draw a graph. *Proc London Math Soc* 1963;s3-13(1):743-767. doi:10.1112/plms/s3-13.1.743 [W2082388593]  (`tutte`)
- Eck M, DeRose T, Duchamp T, Hoppe H, Lounsbery M, Stuetzle W. Multiresolution analysis of arbitrary meshes. *SIGGRAPH* 1995. doi:10.1145/218380.218440 [W2075010828]  (`harmonic`, discrete harmonic map with cotangent weights)
- Pinkall U, Polthier K. Computing discrete minimal surfaces and their conjugates. *Experimental Mathematics* 1993;2(1):15-36. doi:10.1080/10586458.1993.10504266 [W1974956622]  (cotangent Laplacian)
- Floater MS. Mean value coordinates. *Comput Aided Geom Des* 2003;20(1):19-27. doi:10.1016/S0167-8396(03)00002-5 [W4233370650]  (`mean_value`)
- Levy B, Petitjean S, Ray N, Maillot J. Least squares conformal maps for automatic texture atlas generation. *ACM Trans Graph* 2002;21(3):362-371. doi:10.1145/566654.566590 [W3000211808]  (`lscm`; the algorithm behind the SlicerHeart ConformalTextureMapping CLI)
- Liu L, Zhang L, Xu Y, Gotsman C, Gortler SJ. A local/global approach to mesh parameterization. *Comput Graph Forum* 2008;27(5):1495-1504. doi:10.1111/j.1467-8659.2008.01290.x [W2113085766]  (`asap`, `arap`, `arap_hybrid`)
- Sheffer A, de Sturler E. Parameterization of faceted surfaces for meshing using angle-based flattening. *Eng Comput* 2001;17(3):326-337. doi:10.1007/PL00013391 [W2086211692]  (angle-based flattening; not implemented, listed as a candidate)
- Sawhney R, Crane K. Boundary first flattening. *ACM Trans Graph* 2017;37(1):1-14. doi:10.1145/3132705 [W2607599593]  (boundary-controlled conformal flattening; candidate for a follow-up: direct control of boundary lengths)
- Springborn B, Schroder P, Pinkall U. Conformal equivalence of triangle meshes. *ACM Trans Graph* 2008;27(3):1-11. doi:10.1145/1360612.1360676 [W1990120195]  (`cetm`: discrete Yamabe problem + exact layout of the flat metric)
- Smith J, Schaefer S. Bijective parameterization with free boundaries. *ACM Trans Graph* 2015;34(4):1-9. doi:10.1145/2766947 [W2076159000]  (`symmetric_dirichlet` energy)
- Rabinovich M, Poranne R, Panozzo D, Sorkine-Hornung O. Scalable locally injective mappings. *ACM Trans Graph* 2017;36(2):1-16. doi:10.1145/2983621 [W4246041266]  (SLIM; same energy, proxy-based solver; candidate for large meshes)
- Schmidt R, Grimm C, Wyvill B. Interactive decal compositing with discrete exponential maps. *ACM Trans Graph* 2006;25(3):605-613. doi:10.1145/1141911.1141930 [W2053126129]  (`exp_map`)
- Soliman Y, Slepcev D, Crane K. Optimal cone singularities for conformal flattening. *ACM Trans Graph* 2018;37(4):1-17. doi:10.1145/3197517.3201367 [W2810836866]  (principled dart placement; not implemented)
- Stein O, Grinspun E, Crane K. Developability of triangle meshes. *ACM Trans Graph* 2018;37(4):1-14. doi:10.1145/3197517.3201303 [W2810303719]  (`developabilize`: related idea, different energy and free boundary)
- Grinspun E, Hirani AN, Desbrun M, Schroder P. Discrete shells. *Symposium on Computer Animation* 2003:62-67. doi:10.2312/sca03/062-067 [W6963932533]  (hinge bending energy used in the Warp shell model)
- Floater MS, Hormann K. Surface parameterization: a tutorial and survey. In: *Advances in Multiresolution for Geometric Modelling*, Springer 2005. doi:10.1007/3-540-26808-1_9 [W1621614599]  (overview)

## Distortion measures and discrete geometry

- Sander PV, Snyder J, Gortler SJ, Hoppe H. Texture mapping progressive meshes. *SIGGRAPH* 2001. doi:10.1145/383259.383307 [W2097246665]  (L2 / Linf stretch from the singular values of the Jacobian)
- Hormann K, Greiner G. MIPS: an efficient global parametrization method. In: *Curve and Surface Design: Saint-Malo 1999*, Vanderbilt Univ. Press 2000. [W1600753972]  (conformal distortion sigma_1/sigma_2)
- Meyer M, Desbrun M, Schroder P, Barr AH. Discrete differential-geometry operators for triangulated 2-manifolds. In: *Visualization and Mathematics III*, Springer 2003. doi:10.1007/978-3-662-05105-4_2 [W1626653188]  (angle-defect Gaussian curvature, barycentric vertex areas)
- Mitchell JSB, Mount DM, Papadimitriou CH. The discrete geodesic problem. *SIAM J Comput* 1987;16(4):647-668. doi:10.1137/0216045 [W2108567935]  (exact polyhedral geodesics, as implemented in `pygeodesic`)

## Standards (not indexed in OpenAlex)

- ASME V&V 40-2018. Assessing credibility of computational modeling through verification and validation: application to medical devices.
- FDA. Assessing the credibility of computational modeling and simulation in medical device submissions. Guidance for industry and FDA staff, 2023.
