#!/usr/bin/env python3
"""
Bug #0058: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['normalization', 'reduce_mean_first_axis'], ['broadcast', 'cumsum_last_axis'], ['attention', 'matmul_4d_batch'], ['attention', 'matmul_4d_batch'], ['constant', 'cast_fp32_int32_roundtrip'], ['broadcast', 'neg_unary']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0058.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0058.onnx")
INPUT = np.frombuffer(bytes.fromhex("27c402bfe15fa6bdd039c1bec01b07390f9b3b3e35cfdfbfa2af903f3fb9013f863882bea3b4243f054f9ebf77f32a3f47eab43e88bb2dbf8e07753de3c8403f9ab7c6bd82b5c73eda8f133c5d36f7bfcb063ebf35034d3f00a3423f5df4923f35cd1b3d754a353f5f56a2bfaece23bfa8b2a53ea5119dbfb3b2cabc8cbfc7bcdb79f23f8d92293f49b6743dd57b2abff361df3ea0a018c0882db8bfdf248abf7be52cbee51678be9a4e6cbee94355bf805f10c0c802dbbface0f53f5d48103ef73af5beaee6c83f63dc0ec07f391b3fbcacf0be300f903fad67d6bfa64cb3bf24f7e3be456a99bec2df1ebf919503c03c7d44bf0ce153be77e6d13fef22eabdb4f084be294028beec3d1b3f8c59a0bff606ac3d78a69a3f5633a83ea1bad7be6ac6f43e58e5294094f1853fdd50c73ffc4c35bfc18c79bd2d5641bedee9bb3c94e492be1a9c06c023b60fbeb893a5bf7101eebe753cb4be4ad5383f3331603fb93d88bf4d0e42bf0b3b7bbfd54cbabe85e0b5bf44fea33f8e1dbc3d6910c1bf24424f3fcb18483f58e2aabea2ce523faaac323edffe7f3fd40e7dbd551ed73df5650340a2b51bbfefb426bfdf52a43e556737be9e572c3fabf9bcbf8dd1d2bf9af994beccd3753fce3ed83ed4aeb93d3ac3913f1872ccbef2a99b3f3bce6a3eba99a53f607c943fa328be3fea2e793ec850cb3e4b757abfe517d43f136218bfdde8f13e00d705409b7918bde2f5aa3f49cac23f6bd98fbf19eb053e60a734c0bfff20bfb4b3eebea05109401977ab3fddae8bbeaab8543ff99dfc3f0b07173eef1c983fa616b83e3e20d4bf03d934bf1887823e76bc68bcdd8536bfd2c8ffbf74473c3f7c2d8e3e0c25773fa64d093f48ba093fd1d4433fa58b99bf29acb83eed901cbf0956cbbfbf8cd43f4409083f1fb6e0bffe1b1bbfd6fce4be3804a4bf811634bfe0f1413fef1d5b3f163083bdf9ad4e3e3571023f9ae7eabe7f398bbf8b2c23bd994dd0beec3dd5bcb646a53faa2783bfe0cd923fc46a59bfc3810abf3c2889bf14f91f3e8482f73eb56e0fc0debf4b3e89a8de3e859f173e06f61cbd8f61df3ba3092ebfdb79213fb587833ee2f5a23f0e24723cea4d33bee31c29bf8e2dac3ec5e1cfbdd1850c3eb8fd903fd90f95bf3f7704be788cbfbfe69719bfc3628b3f96749b3fe7fdb8bf105ee33eac6c8cbfaf11103f1f290d40ea739dbf13d72ebec199a4bf4dd60c3ce6ca1cbef78a7bbea07fa33a518d403e1d571840d3df38bf238e82bf95c24c3dd625cc3fc4f059bfbc51aabf678b45bf28594fbfde6283be981bb5be8aed8ebf690ce7bf790a883e8e20f53f4a7770bfe6922ebec244e13e4bfa423e1401c73eb9e5a6bfefcb143f03a3ab3ffa6ab9bf495d93bf60e7983e7ff71ec0aa37b9be5fef003f7021fa3e30df86bd564ba0bff115f5bd30948bbfa2e5a93f614db53f5508983f270d103e94899dbfc67c09be6d1797bea7c0e1bd6c41b8bcc94d7bbe89c28c3e764a18beba6d38bde9e65bbf3bf2563fdc10543e05fb8abf7bd23b3f132b473f80fef13ee1f7a7bea22ea83e67c7be3d35e9e7bf5021c4bfd3493bbfebac9abf4fa2efbe2a0ba6beffb3a4bf7181fa3fb12d0dbf808d6f3e1a0533bf61f190bfcec9eb3f8e82b23faf9ab0bf6840f63db8a4343fa13096bea4be3a3f549429bf09e162bf6b6a02bf89d8343f04bb423e0f505cbed0afd0bf9b21a03dfbf2a4bfee3a0abf2c7c54bf4ec2d6bfb01930bf5ec77a3f9abbe63e836c0bc02f94843fc19803bfc31f783f1ca447bf062822bfd38074bd8e44a33f37d7f63fd243a93dc6bf683da8ece9bfb67e29bef4634fbe1118e83ec2a376bf4a32dd3d57a197bef095c6bebeeaa93ef6167dbf182853bf7cb7f7bfac01b1bd66d5053f2fb9173e39d5f73d6b384f3db37062bec8a253bf7ac4bcbf8f45d8bee66722bfa3ad3bbf163317c0234ba4bf04db013f02f866bf6daa2740e80be63f5df388be8297e93e5d557abd8d1b47bf85e697bf1d79923f53337a3f9e36b23fe1e618bf00e868bce2fdad3fdb58f63eeb4c45bf4dc6723f8443483f16cf02402e5c073f1cea3d3f0fa4243ff90db43f39a8203f0e33a4bf9208a53fadd9f7be4188903e29907cbf181dac3d59b591bf903c52bf6a5a28be87a0723e4f57893e6189e7bf02982bbf83746d3f745b84be46c1d4bf65ad443f2149f2bf0be3d63edd9f3a3f84f4b3bfe97bf53efe6a9b3fff17c5bf64b5b9be1978f63ebb10603f5c899f3ec5e3d93ebc4836bdcb4a1e3f9f6c64bf5169e13eb407babf02a5dc3c3602c93eecd633c00b06543f1146d2bfa0dfd8be04d3c8bdb2ad533e98c93dbf828e2740bb54f3bfe9cb92bf50945a3f9fb1bbbf48020d3f88a2253fc381043f1f25933f3f9343bf130c803f49ad9f3e30b8023ff0ff0d3f2597e73efad729befbb42cbf0263b53d997c283f2331d7bfa0505e3d97a0c0bff1cd34bfb1fd983fd065d53e8535c83f90a43bbff1692d3f636b293f4bda8e3f6cdae9beaac56fbea3eda03f6819c43f6d768ebfa91c1bbf6559db3f6fa3d03edb0e5dbd30aea43f6e36fc3d72c38abed040b1be298a01be5e688ebf1ecb893f9e486e3db97f92bf0156d5bfcaad8ebf4286dabe51b84b3fd89f82bfa70839beac19aebc99e0bfbed4cdce3fe070c73ed34d8cbf6874333e5eae1a3fae26ca3c919df83d61cca73f3316093f63071abf5878c43fd739cc3ee7cb4b3c51a8c63fa7f16fbf99dc86bdfe09ee3e4ee854bf0f290abf3dcc963f93d2b0bfdfab9cbe9cc02a3f50849d3d8771d13e4b3232bf4672bebec9c400c0104d3dbfae32edbf5fdadabfd784ba3fa9f76f3f8d48e63fd9bd7f3f08c68d3e958a04be05c7633ff4cda5bf8472acbd99f5fabd6e7cb4bf13ea8b3fd27b6b3de16760bfeee7973e532720bf5b6a063fb4d682bfd41a92bde636b9bfa7840b3d4300703fb2599abed5f3813f311a1bbe837e8abf15c56e3e3a7a723f96f365bfc7838f3f1426443e3fd47e3fa96a043e9fe7e4bf6e6fa53fc6d67bbffe14adbe3bf605beb066f6be3fd114be5230423f58e68abf93659bbf480e533f0038a0bfb4837f3f01b4f4bf2a1504c0d8b78d3f804505c0706d7a3fd42ec53f95944fbd75383c3cf7c41bc06e367bbfcd75693f8e338cbf6f078ebfa13bb6be3e9ac43ffcdb23400d85c8bfbb498bbf01346fbf6e00b83ff5502dbebd8ba2bf292edf3f9731a73f6249953f1ff802bfd85bc7bf21c29fbfeb84e13e524991bfa03936bbfa99ae3fa3e108bf721eae3f803633be2178e3bfd559eebe79d306c0233df43f9bf427be7f4e03c0983785bf2679efbea0212b3f2db96e3e2de9563f227bbfbe01b201c029d3ddbc1628cbbfd8779b3f065ce0bf9800f3bfe601453fc8a09dbf2320003e2414e83e512ba1bff07a163ec9c8d9beadaa09c0ffefc9bfe2a5c3bfdd00bf3f30a5293fca19f5bff1076a3f4d789c3f43bb643e9f74edbef4b0513c295033bff83d1f3f4e57c23d5f4c06c06ff2843eee86ee3dd2b2813f06ce4c3f59f831bfc1b98dbfaf7f533fb1a65f3f9781c23ed2c134bf61fe393f0baff33f0e5c07bff9df07bfff05c0bf45b68d3f43cc623f0bcc593ef80ac8be6a0ab6bf55f758bfbadb8f3e9603b8bf7e91bdbf652a34bf49c54fbe0b37d53d886d95beb688dcbec7fb003fcf8335bffcbd7d3e89f377bf57f57fbf1c581dbfe4f10dc01a9dd7be724a3ebf09eabcbd4846c83ef15e72bf904e44befa46383f816bc9be51b7ffbf0654de3f4214813f85208bbfa3e9c0bfa5f8083f491d34bedbc6ab3e8299d43e243440bf6157db3fb3aa253fc0490b3f09d4843f87bd78bcfe980d3f425a263f464ac93f8a0a0640cc4b603e95093f3f24ae2fc01375243f711f473f90dfb2bc2a7a723f7d7adebe814df73eef1a02408ac0103f9b395a3e0ccb2b3f3772dcbe47c92e3f9fc2d9bd44247d3f7713d6bff3f1ba3d8cde98befc2edf3f7d11583f528c95bf35e241bf75e4104097bb3ebf3747a9be28e1a93f8d8b203e1093bc3e8f0beabfe84c213dba289d3fc380753feee00dbf3c3af9be31da99bfda8a053fab505b3fae780fbfa89337407cf6f2bebe2c86bd8fe8763eb67eacbe3b7f8abfd290aabe895b4abe5b579f3fad911540bf1dce3fd5538ebfb3821cbf9a6109bd754976bf4a25ad3ddc31d23f46f3243f3a1e22beab5fc8be852589bfffef5bbf5fb6c8be0186813fcf483fbf17fb263f9e6d0cbf42d7a33ff1cd84bff4f1ab3d1425edbfd7a984bf4f17783e279fb2beda61313f2d619abf5e0c143f84f00b3fdf0a13c02d4a69bf492a163f16f7b7bf9913ad3eba88d5bfa3b49fbfcebed7bd608d483ee3e203beddd298bf5a0a50bf666688bf398ca93f42daa2bfa41e853f6535a1be0b6ef83f526dc73ed89a66be58e7debe289389bfa3f4993f609f3b3e419fb1be314a8cbf46760cbea4b324babd1ccd3fec4e0e40843b37bf9a3ca03f92da3f3f36c0e53e49ab3c3f8dea91bfd7a7fbbdcd1075bf6f7afc3e335b0bbea6ddfdbefec966be266447bff47f27bfc45eb0becdd45dbe41ece63fe8eddcbe08bbd73e2dab8dbe1f971d3fb0b86bbf6cbe88bffa56843f09bee73eba0bdbbe02bac43e865e1740be9d3b3ffb64333fa685fd3f46a6a93c9465dabf1c738d3d26ab9fbfffc1b33da531e53e81c433bf5e07afbec56d94bf336381bf4cfb873f823a5f3f8dc7c0bd035accbea754093ffe0afebf5eaa0bc0ee0dc73f2c754b3fe342923f2505b93faf39303f2091a73ff99a4cbfe077833fbf25873f6fe7a63ed171143f9a6ff33edf95883f11b9363f178fcbbe69216a3f8dde89bf0e6d3abfb53b993e47a4a93fb42724be7e4b1a3f0894cabde4fa553ff09d12bd98a732be29070bc003112e3f0092153ef8eeaa3e51a2503f497ff2bd155ae7bd1312423e92eb14bfc037aebdff0bc1bf6ed1fdbfed4fb03def1a52bf627d3dbf363a943fe0ca173fee2dbabfe6b04f3fb21ca03f73c1c7bebb2383bf078dd8bf7cc33bbf9c3eea3ca9ff433ff8df653f5f1d863f9ee888bf17848d3fcb5f3ebf7e16383f19afadbfb7859abdfce8493e69e1e83eaac692bf94c8403fe3826f3ffaebeabe2cee443f1604933ce94ee43e7d31213e3018ad3e34026b3e6002babede1cbebff7d7b0bf666571be659d5a3f608f1dbf7a6c3d3f418ba53f566053becae1ec3de3f1cdbe8bf69cbf60578a3eb2d144bee7c7603e1fb9073e2a0472bc204e07c03f75763e34c73fbf04339b3dccfac8bfc71c8b3e1080ee3f813526bf9a20ac3f7593863ea22f40bd5a9d303f6531943e7b9a7b3fd61eb9be57eba53f2e603b3fb6235f3ee803423ef85c05bf78d80bbe1d6af83c7ee8143e9d7f993ea6171abf2d3a813fd3b03f3f91a6f3bf3d8b843e2040a83fcf37ad3ed871633fe34f9f3d964e65beb67e4d3fb2e8273e308b80bf059ca23fff025bbf172435becf50173f2f68cbbeee34b63f3b9250bfcbb6a1be0a49773f22204abed6700dbf9a8c013f8f3c91be281b823deb17033fb308cb3e9764a5bffaf74e3f39637abe2fb8b13f3851d53f7b3363be77a2c8bf8575a4bf066ab23f23c984be4c6656bf75e9b0be69059cbf8b0596bf15d122bfe2c2a8bf5ca8abbe9d8c1cc0df16aebeaf97fdbf114203bef5da00bfa833eabea7d7903ef37332bf4bbb923eeac1ba3e73e09cbfe91c623fe39c563f414e2a3f78b9853fe2d6b3bfec2f86bed73cb43f074404bf8394a13f4d38d13e57f00f407db548be7826823e1a50213fca96973fd544713f163cb0bf53a31240bace5d3fa4e537c053bf1d403902563e2a9d83be06679b3fb215ae3f2ac438be25c5643ff34e73be3845e7be5a67adbffd53d13daf785cbf06422d40de9e843ff298fabe567c40bfd683a6bf68fe2b3f3ffe45be235388bed43459bf2796893fecce2240f02ccb3f736b79bf8a90c9be8ab6703f0bd4ee3ed1d0cd3e5baa18bf26a2cdbe4477fd3ea3056f3fa510643fa93f2a3fe9ffdabd70ab0b3f58f8e6bee817debe25d6ffbe7de346bf371b483f0f14b93f296484bfa74ac03eba7f693febdfdfbf4073493f59cdab3e9b4a2e3ea1862c3f66e6144046a338bfd6a41b3f0367963fe7a3f63e8600a1bf20982b3f03d1bdbfcb5ea8bfb632073facad043f9fdeca3e5eb98b3d0a8aec3e5d94e8be28a7923ffc9ca7bf18cbc7bf4a5260bfc800143f5de3db3ee7e5a7bf0076acbf37aec4bfa2ab98bf78ff83be295cfa3ed0b535bf10c4bc3c16f5a0be418967bfc0dc3e3fe61ebe3f2026833f928d783f33ab81bfa918783f2dea633f2dfc923d3552243e9567a6bf7e67003e554db3bfab8328bfef6f3a3f5ad6963f1741a23ec80d63bf88881f3f72f68d3f928bce3f3097773e134aefbf4ca9e7bf757e7ebfcc699a3e132482bf86b7ed3ed4d2693e5f03c9bf461a913e781cdd3f515b0b3ed81872bfbfa89ebdf09cc6bfc8b385bf5488c4bfe337df3dc30519bf7b6473be569655bfbde119bec9d83a3e8d89853e2153373efdaa004031c0883f29d0893f705358bee425c73ee839423f4c7ed63f5e8c99bef5b8c23e546ebdbe6781aa3e893e623fc2d897bf5094653fc186503fd70eaf3e1b925a3e26bcd43e488d27c0cbfd3abf7f2db43e75da123db3f8803e3f1fae3f695220c0aea872bfc1de2abe6bff323f29d1ff3e9ce1b93ef5f84a3fe9c1823fd3b4e5bd61ee9bbe7da2c4be977bb93f2dce1fbf80cf22bfa3dd78bf65adcbbe7d135a3f50d63d3e80705c3ea93804c01dd7773f125cd33f609a95bfd7fcabbea899923f04fe6fbf910e333ff6ebb9bf1491123f8664f33da197783ea4a4203f64870fc06896023e063484bf92407e3eb357ac3e996d5e3f2d2027402a4dbd3e593c203d108332bf708cdf3fcae162be87bc5e3f27dc753ba3ea3bbf9edcd6be5b4980bf21e4bcbc32ab01beb7a23a3f45cf96bfd77839bfb50f20bfba9e6e3f154994bf043bcb3d534173bef064f43ed44009c00cbeeabfba6247bf6f12ae3f580f90bf7b1f483ff06da23e18bb953ff953b53eacdfbbbd43e46cbfea3df93f0a0f703ff281e6beda2bc93d6d2242bff9f0813f0a919bbfee7e73bfe4a6a4bff47c983fb19303400137373e830db23fc238d7be34c69bbf33f3583f02ba83bd9baaeb3e43729ebf8487403f72894b3f9fd16b3f5b6c33bdc3b98e3fdb78dfbe0e83743ea3e3b9beaea268be5e5a0a3f80f9783f332ccc3e1a4135bfc896803fae9abf3f38cca73f81f2573fb307993e34c9853faa17523e7b0be03eee0a97bde2a1bc3e5747d4bee8050dbffdaf2a402e4e35bffc2af0bfa93a7c3fa214cbbe9eab623f55b9da3fe0baf53f96f1df3e0854aebeb3bb8abfc9ef903e16fafebc9cc2bbbee7a00740ead71a3f758ab9bfa34f59bfd12d093f172b9bbe9dc51cbdd13493bf9bdc5b3fa8cc47bf28f52e3f9ad2053ee2bfdbbe01bfc03f529f48be7e70f7be622b6ebfce6061be57a0243f520291bf1363273f32f8723e6d04493fbba547bdb7e6c33e62f751bf0a6fa4bfd673c3bea5206fbedc6e55bfad3419bfd553c03eb669f8be8a37303f7a68983f7be1973f98fa31bfa5cf2a3f8551e13e27b555bf5c9a423ff581c7be78de1f3ff956d8bfb6c434bf5b9310bffa64193fca5ce5be828b7b3e04233fbe7fbb2d3d6515a43f204488bebfc0943ea8e5ab3fb91cbfbf0b571a3f6e8b15400df9853ec597ea3c7931413dd31b01404aa74f3e4f246cbf50dd333d16ba593fef86a3bf179be13e1cab1f3f51f2733f2f8a5e3ff9642fbf46a8c6be488b423f1cdc2fbf9afcd53f4cac993f380ea93e43d499ba5952ee3e6f1bcb3fb80ac03f19d6aabece524b40f8ee4bc0e2492340c8c1ed3f40708f3f83bde53e3e58033e3ddc82be2b7b353fc6c08d3f1457cb3e9253ff3e9f307b3ff269d33fc2799e3f80560cbf54f8963fd362fd3f853f9fbf8f5fd5be6ebda4be4597f7be49c9353f5b5385bf58c0003fc4cbadbf2bf3b13fa96abfbfd442b9bfd0607e3f6931843e1a6c8ebd3b7ace3c3ec9aa3ecfed2d3ee8d8ca3e87c7a23f2aec9ebe3106b53ff41e553eeadb04be72c868bf70f38c3f3acd41bfd4c2a73e0eaf8ebf3b447bbe090c00bf28784cbf7c3acc3fa81eb6be06676dbfc4d988bf8f6c063fb8ca6d3d76fdbbbc849887bf288b6e3ec63d09bf634b1cbfce7fca3fc9b5ac3ee60f553e10c8eb3ea4a32b3f46a5813e49bbb4bf3a827e3f05b151bf8513bcbebe9448bc20861e3e4f552abf11127c3e6141bf3e11d47cbfca97dbbb76f558bfe68a7b3f92c7afbe80a1b43f4e4d05be3b3b793d4a69a7bf8496a83eb5b873bfd5f9603f69f58640370a243d6bdb9e3e096a97bf1d59a5be780c15c02fbcc73f204618c048fe8f3f37ee743fcc9615bf3cfc1bbf96c5c6be5a290e3f1877f23c28219a3f4a7669bd646a8abf1f24843f790b99bf7de4523f056f893fdae3a1bdd2e096bf568896bf4d537f3e736d77bebfba083e86b6adbdaa2e11bf57ea5c3fd1db493fe96f733fdd4ef6bdf32a1f3fc4b7d4bf358e1a3e0bb6863f5cf29cbf827c3bbfe52f3e3fab2167be3cbdb2bfdcc8513e8149f4bdf2ad903fbd568d3fc07916bfe9e1df3f3a8d6f3f6a06aabdd9bf223fe22008bf4fd75dbfb51dc73efbae52beedf1223f88053b3e9b2c533f9127d5bf174a90bd49043cbfdccf8c3ed87d103f8e5e963fbcc693bf50ca70bf72105abf12e72840cdd23bbfded1bdbf1f6b83bf9da7323ecf99b6be3ae558bf975ea33e347b12bfb1ed113e2a9e07bffa0e2abd7dfffc3f8c660cc02548a6bee34aafbecd7d0a40f0bd30bff35f84bf90b30cbcb09118bfd0d82c3e883c2abf464945bfbd5082bff4daa4be840ae93e9d5a28bfc448ab3f9dc6803f2a32fc3eb8a745bfe56d0abef7027e3f23eb05bff62370bf4bc85d3ee2abb0be8f65a6bf42376bbf625d023f4c9996bf04be443f8bd2313fcf5840bc3d672c3f5462153f8bc7003f661ba73ef014063e1a79893fd73079bff6b2b93e3f2f40bf5ae5fcbf1486503f7ef7663f4019a9be3fc6c7bf1c5fb03ee6ece53d6324efbe302263bf6819a7bfd171cdbfc16093be20dbc6be61729a3e7142c83ff82815beeb0b2840b987f9bebadea33f4a17cabf8254473f3386793e8deb9a3e82518e3f10db173fcdcdae3ffcdbaabfeaf4db3f015ffebf298dae3d2c3d21bf4d7af33feb6d5dbee04377be8eb4ed3e7fccab3fec6faabf64301fc0d44de1be9412fdbf157da6bf7d9d993ed0fbc2bf754c403fe5ac43bf4069ca3e37d90b3f5347bfbf16402e3f985dcfbbc280553e7382b5bf3449993fcd71bdbf5ae3a2bdeede23bf37cf463f9fd3a8bf3c0a543e6f05ba3ecb3c36bff508a0bea59d97bfa31e8d3ff275b8bf7e55dc3c37b782bfa64b793e6cb2153f3bc379bfdfa69a3f347560bf6c369ebfae719abc3fc21fbf835841bd9ae7c0bf9e399e3fb3d4243ebd8f903fa40b993f4d4d86bf7a645cbf2099ad3f2ee4a9be6bbe66bf49c998bff4002340affb6fbfa814f43fad468dbfcc6976bf6f1cdebfb8da86bf950a0bbfc74d67bf19ccf0bedfc277bfbdc38f3ed4d9eebe5d9f9e3d009817405129c2be321fb3be8819f8becc87a43d0375d5bec65bf1beb4e9853da6d2e3be3df280bf8a51ba3d5b3ad83e1fae96bd4570543f5ed71ac0000fc4bfb4ee2ebe2229c43ff978423f46cc613f398fd03c478609bf84c3c1be1133b7be21bfb7bf706997bf37554fbf4c32f03d3a0c4ebe1f68613fcd13bebf0e1e023f0a1fb8be9fcabebf81313d3fa1c0de3e6a0005be20b9e6bf415484bf359b88bd4ce8e8bf35f9d93f5ab0e1beb95f19be521e4cbe22911fbffd2e8dbfaec6573ed8c941be6b4580bece0392bf1e08793dd622f23dc5386b3e8e8b53bf05271a3f5e7ad5be3905b2bfee63f6bfd0fc10bffa2a67be6b369b3fe6a89bbc1a62663f78bc1abfa9ab403f0412ed3e60524cbf640c5cbe1106f1beef76fe3d2fe720be18172abf145da33e765c153ee27d04bfd7d37d3ff21650c0744f2f3e1e2809bc874af3bda95e7dbfaaa2a23e346c6c3eea3cafbeb8985bbf631680be0349acbff9fa86bfc095253f3bd3dfbe581d9ebe111415be44cf5d3f1b2de83f391ee43c6515f43ebe784e3f849c363e95ff453f10a0c53f656e343f372181bf47d89dbf18a309bf86dc953f40baadbf09a8afbf1c694cbe08a6afbe7775543f76b932c0aa66b7bc7703a2beed1017bf5868a73ebb71f2be0434513e5eb7debefbb28dbf2774d83e70ed8a3e98f04ebf3698ee3ec92a01bf03022abf05f0d8bf5ba6dc3de1b443bf861dcc3f015411bf7ebde2bfd489493f8f691740cacd11bff3149b3e509e463f1f4432bffab7163ef9ad50bee97d9bbee2ec843f269d8dbf60e37bbe94cfcf3f028e8bbd71fc893f2ec4dd3e589646bdfe215cbf49ef95be496303403dfc57bfb5a0083f059bbbbedea166bfc79e18bf1fb7a43ffc50f6bf1efd6c3e227c6d3f6be4863ed20016bf5766c23f589be03faa7c94bfc3d97fbf3d07133f49876cbf539f003f3fe3fabf1c05a8be6ee891be3bfb373e9a4a26bee10dacbdada5993e984e2640d70118bf0cf6d33fc03d70bfec5753be44974c3fae70d3bf834a69beb024efbf7d6082bf5fe5783fe483043e38c935bd23eee8be0291e9bef28bafbf8243fe3e85ab09bfd68d8dbe3606e33ecec8f73c03ea813c3979833f02fe2bbfc682ef3de445653d70a467bf248d283fc6fd92bf2b6e1dbf75ffb1bed83564bf625b4b3e5cfdb7bebf1b1cbeaed0363f2e2371bffa0038bfeff336bf102e823fb4f8ae3f5f5dc6be083bc4bc425826beb6bfcd3e17013a3fe0c8a2be6233fcbefcdd40bd4199983f76756c3e05e97e3ffd8a7c3fb8f0f73ec5191d3f3e30ac3eb05499bfe6accf3f0041f33fc365d53c0229cf3ec8e78f3fbcb7fcbd186911bfe863e03ff3317b3f8ed89c3fa2cd39be97eded3e5307c93e558da6bf11036bbfdb34c83eb148cd3e795e47bf785d0f3f79773cbf32351bbe53f3df3ec330fcbf54ce013e840a3abf11f6d83f32e21c3ecc19d23eedb450c0d9a1a53f12e58fbe8d19b13e948d9cbe83c83cbfa385493d2ffe67bfeb6dd93f06b2a2bf46cf06bf890ab93e21fce6befc8a48bfc0ccdd3eaf37d6bff1a0ac3efe3cfbbfa8e0d63ea18e443ee6e20cc0bf651cc0df1d76bfcfbb353f8368773f8c55c7be7596f6bedc7db1bee52eba3f3b381d3f713416bf53d867bfa3c3db3efe72bd3dc78dfcbf741b76bf26eb17bf7600393fbefd0a4092e2a3bf4ba2d63cfd7a2940a394a73fd1c0aa3f86303d3e55417abf6b0f2c40f1d686beac5c95be06ae883f4477163dcc5b5dbe8e0598be9cbb773e10d99dbfff67133f297087bf383c3d3fc1bba23f3dc45e3f2faaa63f454c1dc01e78b1be45f1703f465e1c3f72f47dbcbfc485bf540fdfbe20bb2b3f205efa3e5df4d93e4653e5be23d04dbfdd30eabe0308263f8267a03e44b5793f33099d3d28e0adbe2f29a4be96bedc3f27be1abf4d13c8be17e81d3e996e733e4493ecbd680b83be3f9b4f3f1b67a6bfc3ffabbd19308b3f653cd0be54b5973f0745e7be373252bfa5127d3fdd8f22c0b7da5abe5e0ad03fee7f933ed870d23e8e7f023f025193bf3d97233fff89e4be71a0ebbf187d36bf9fed243ff4df3540bd72003dff8d013f47e4533f33945abff25d873e4884f5bdce4b3cbfec1087beb85b23c09a4dadbe6a26e73f6c01a1bfe5b30c3eb40049bfcee6a5bee19cedbeba0eefbeff852cbfbe8125bea57609be020f673f0cfa84bf28b0f53f2e3078bd9b9ac9be96de91bfd8d525be71ab833f64cb1bc0959125bf0c2b9bbf7ae27f3f5e7d16bf52c2793ea4b4a73f8ca0c0be459d8f3ff8f7463f2b1df03e9297283f8c1d04bfdb0d81be6a761c3f658b4cbf89c3053f6b28c7befbd2aa3f746486bf853c2cbf8c492b3e899685bf07b47a3f6bc9c13f4f600abf238770bf91c130bf8b1dae3f93bef33e09dee73f8fb4dabf846d35be017baebf642ca0beef6724bf4ac3a83e5cdaaa3f1ff5933f86f6433c170fcfbd511612c0d86408bec7fe98bfb565043e952e5e3ede5eaebf0f6c64bf2d2b13bfbf0939bee01dbebf577ecbbed8f4a5bf0c68b0bef248b5bfe29513bf17f4c43fc6d973bff52628bfe3d6aebf57fa96bfb42b593f3817633f38e0cb3e7ba2343f3205babea4be00bf37e3c93d3173c1bc5f2f843f3ce879be32d5af3ebae8213e758fb5bf09926b3faf86123f9a250bc07e528b3fd0a0873d51c1083f8cbf54bd1934783f6071ed3e2c6fa83f8b76d0be866b47bf7f69ca3ec9c37e3f6da605bf3d6effbebbcdc0bfd178b7bfa343e6be5974be3df1f3febca87a25bfdf64bd3f4efbb63f6fda963ff46612bff7af8dbe61d40c3f7a02913f057ba9bf35d584be8f3ac33ea1b4b03fb70a003fbe23c63f0eb10cbf4f33c23f89fbe33f741f0040d17827bfe50887bf7052ea3dccef45bf0d42d33d5dc621bf80661d3faf45cabe95745fbf902f57be3bbb2ebfa81eb53ee9613d40e280903f9f8d7cbd90869cbff6cfda3d7fef233e4f7e88bfa75be03d4bad5fbf653af5bdeb35583f927c9dbfa62b6abe5ec77d3eff6b8e3f36c4c13fc608a13f159fa43f8dbc6ebfa92002bf881090bff978103fbc55e4bea81170bea824023f46a60b3f714f88bfaa94ad3fa949cb3e5aa2ec3f8de282bfc8ab1fbfc592523ef62a59bf4509453fe0179cbd84a8cabca2ba583f0ef80a3e4f28b33f4b12123f14e51d40036baa3fb02a8fbffe17f4bfcdbb83bff40c8a3f06f109bef121b23fb17d173ff03b42bfeb742dbf427913be5cceb73dd656a2bf275c1e3f12118abff3b00dbfa207da3d6687803f7a3a5e40f5768bbfc8da4d3f1ea7bebe66e16abe978a48bd038b863f141e95bf3655e7bb6b39b43ed6fdee3eb819c83ed31d923fe3e2dc3e548d64bea45d5c3e8342f93d1943f73ec13e493fdc3d20beca79fa3f5577c23e18e5bfbe3be904be900c39bf0ff837be9efebdbec765d63d09258a3f6e5bb8be008205c0f7d72c3ff99eda3e0b9181bffdd5e53ed77f7d3e46131140e653983f37d8d1bfa454123f4554233e82a881bfa267983f7822553fa2698bbfa12f0c40e274b63f6799723d087b50bf68a8a23e2c51903e1d8791bfffbd1bbf19f7014046f9a0be8c03f23b39e041bd227dedbe3466a0be87e569bf7d6514bff9519f3f38b1aa3ef627283fdaf9673c3d641dbf10de19bf41b437bf219c413fab350e3f126ed5be4147bd3d163576bf16ed523f5293883f45f2813eb023053f0ee5253f8cb1fb3e9b8c3b3db9083fbe2631e4bfb4a85cbf5eed2540ca16babf60c10abe145a3a3eca84ab3fea185b3f255d89bf466861bc74b9a63ecda5f83fa5dd07bfdd4004bf5f4ccfbf1555b23e3e4bb03ee1fe643fa5860c3ff23af2be8995f43f03e0ff3e88c523c034f8d23efe9482bea4600abf2a590ebf225a9e3fa9c3283f50f4aabf3378893f6d343e3fcddd5abed87225bf0eaed7bf76f7bc3d8cd256bfc60e0a3e662c1fbfa918043eb9560740ec60c0bea49d80bf384a88be5566073fd81f3cbf7e40bebe19338ebf70f614bfe710a4bf690f104022205dbf708d86bedf2b45bf770d763f3056d3bf4aad5bbf34cea63ea71519bfd175913f95d2673fde65b33cc046013e755e224062a3d7bfb96a1e40710bb1bdbb3606be9dd05c3f6a56733fccc666beeb47a8bf040f6c3f1caf1bbe57e4a6bfb7d4233ef7f93c3f821394bfb770dfbe7950923e03d95a3e1bd26d3facea163e321be1bfbe38b6be5976d0bf61d4fbbfb111ebbfc1caed3ebaa7b33ea0ff3c3e0d828ebddec583be2dcdee3ea18537bfc53501bf84b6c43e3b7a21be79213f3f910041bf71d9e8bf8ab902bd403b45bf58f9f6beb224d9bed59b2bbf5cc43ebfe43cbebf54aa8f3f06e8e83f8f9d1dbf2b47933f67cd6d3f4d3daebf41f877bfabd856bf10bcbdbe8254623da0f53a3fdffd073f23c2d93d5b2ad43e13aca0be4a527fbe7c5411bfca2481bf34c8ff3f2de7933f03e0e23ee04feebe7d7bcb3f251796be9348763ee1868c3ffba7a7bf5b698b3f0857b5bea858ce3e56f6053e11b6af3fa6bee1bf6cfd303ef169be3c2cd88f3f4eadd73f3701e73e1f1e82bfad879b3e85c40fc014139f3f4c502ebe91f2c8be0355e53dfb1083bf4fbc82bf02fa84bf98ae4bbe16aa87bf59f9b83e9208733f4e1995bffe8719bd910831bfecd411c0094b8b3f8a008bbf909405bf1b1cca3f6cf3893ffc6973bffadd953f44b43f3f807e28bff1eba83f5506e23f0ea62b3f8a4bf03fd202433ed2ff56bd3c9aaebdc0f5633de7dcbd3d41374bbe26b302be1462cb3f49fb933f63c872bf291f9cbf8269e83ddc582cbe71b7243e55a4b7bf092a163f7a5f7cbf2c8a8abf2ac3833f05080540c233c1bf48f137bf78f628beb70d19407668633d60fb064047a4623fb491f53e1cb30d3f91902b3f4fea0a4077e382bf9862e03d335b933f2995b1bf097eebbf310704bfea3ec43ed9cbe93ef9f718bf98d9a13f3c40343e5ad43e3f17ffe43f4c95573da339503f52694fbf4816b3bf23fd863f11f9453f6be8d8bf8be45cbf7e79d63e92b75a3f34513c3f4a20ae3ff7775ebe8234afbe7dc8f53ed8e9bebfa52e86becfafa63f15577abf1434873ea75396bf88ffa3bf943307bfde0965bedff3483fd220283ee997a4be8c6d08bf5037083fd223e43f88bcb53c4a05e43e3cf9b13f9d2bf73bc90e91beb7a94bbfd7aea2bfd2fcd13f11d7c73e2e5cbebe5e2b12bfacc9df3eb460ee3d3cfd9a3f31594a3f50df08bf16a6183fa262d83d46c95dbf2c805cbfa0f46f3ee524133fa8c21a3f559c39be05c1d33fd7930ac0451d293f162ed13e417a673f0ef4ff3d8c5f8a3f8bec08bf3ab947bb9451aabfdcce2b3f6b01b03f9820913f46fdbcbdfc354c3e0edc94bf79dec6bfcec0b1bd5c5025c039b2893e737308401e7764bee2a2213f474f993efb5b9b3f73bf033f72cc96be865bffbfb9ac8a3e2d282bbff35d553f2986913fd5580a3e671db33ed81fd0bd7571a5bff77529bfb49d3abf056f70bf412a873eb497e93eb80318c0306b753f59477f3fc767b63ea355873f96940d3dcd2c54be1295d5bf7083c93fe13c083f8173b9be52588fbed3a8c13faf8fb1be5f409dbea6693ebf3eb304c03903643e16dc813eda48e8bce8e8943eddf5203ff4fefd3e604641bf4c8941bd7833b7bf1ef3883e8fd96dbfdbd7c5bf75aed63fa07cca3f42788b3f8e21343e9d33efbf1b7e74bff3f684bfb276fabe3d19ca3ed5df6c3fc4008abfad7da7beb0d90dbe8304c93f0d1c99bff19dd7bf442ae33ed222debf8cbad2bea8b4833f57ba043fd2e7ebbe1a8c06c0711512be927ed3beebe9b43f87486bbf4ebfe4bf17d422c0d4a11abf7362e4bec1bfb2bfe374733e9e37ea3f9e1c0fbff9ad95bfc6e711c02baa2bbf160a12bd842e4a3fd6547fbf608a263f6c9f34c09977c33f55dea33ea2f61f3c1c246a3d1c60813f80f55cbd3f1129bf9abe583fc93b883f7dfc9ebf6b28c93f1bdabcbfa9afbe3ffe50803fd018083fe94036bfe4539fbf221f0bc04f98113f6097c13b7459bdbef5529b3fd9a73c3f2482dd3fde58fe3e970821bff1a4eabf93a5e33d275a5cbe68e897bff314c5beaa84debf0dc9cbbe5567b03fc0f728bfd1b4d5bfdaeb6b3f4d85a2bdcb9c0a409aff6ebfcb32b7bfb36004c02b18723f37a0dd3e1fdf953fc9a81a3ff8d5e73ceffcdabe994167bfdc94f93e69ea96bec6b002bf00eb77bfa8e94e3fa29320bf165b413f4dfa063d2707983fc5ea8e3eb8fca0bd51bc6dbd585055bf3538b6bef1ca0840404df83f89e5ad3f2e04783e40ee63bf680fa73fc3b4953f4475283d78952e3e57a302be5321223ed99c82bf2a0060bf9aad943f4324a33d8b96e03f383e1fbf2402773ff8e22d3fbb1fb2bee6bb743ec3c863bfe41a98bfcfd1703c5fd5ebbf8e7d8abe291441bfc380923fb99f02bfba528ebe23214f3f05866abffa44aa3f498548bfd96c693f4c1ac7bf5d9840bf794f1ebfc314433f5402b1bf3b101c40cf468abfb4e1ef3f4d7670bfbcd81c3fa411083f7867c13e59ef12be58f42dbf56501a3e1590b0bf1ecaadbff40fbabea1ce93bd313b013e3d94db3f7334d6bf55ae293f3d685fbef909b6bdcf0c253f663bb13fbbeab03f6c44febfecbe9abe419a13be3b84ce3e3fab33bf1932c53ff7513bbf67c8b33eb644ca3d2dfa9abf615bdbbe2a8b943f299d10bed3b8dd3fcc0a63bcf5cee5bf108fc0bf894b98bf00c5303fcdb2a63f0d821fbe68ccc3bf533bddbf7f6dfe3e27fa0f3f536f173fd99049bf7fe519bf593ae2bf540a8a3f21eebfbf2e4849be0eb6333f4e38e23ee7467ebec20fe03f1364253f04e7bebeedc45abe13ea523fe4df753ff8e693beb77f1540c8d982bf17f66c3fd6fbcbbe7d541c3e092e6fbf0ffc483ffee9493fdd57a9bfb768b7bedae09cbf1e350c3f74873dbfa52bb9bf73e1fd3ff2eae33ecc184b3f4b2b00404743343d1e61503f7f204c3f1d7b9bbf9b1a4b3e084720c07ea4acbfd9add23f7bc0a53f6c903fbec763913ff19ab7bdeef8ea3eebdd853fccd7223f9e9fcdbeee63103fcbfe01bc5a4551bfdc659b3f174a51bfe8eea23e428c1c3fb6bc493f6ef6db3e2a68d6be052575bfa6b856bd2080923f0f51c83f26561e3ff500653fbde504bfbd7725bfe98e1c3f517bf2bdaab02cbf9a79cebfeee837403f1502bfa1c837c0d3b9103f0dda99bee3459d3f0c1f8abfd4e7923f5cb5b73e681f0fbecafca03b57239d3e815e4cbf69e1223c67e89f3fadc5193fbff3963cf638873fcd38f13e3233e7bf5593343fef2d1bbf31562cbf9140983e975907be9093bdbe1e0abc3e8e61573ee803a4bee07e9a3fad12233fe5a2f4bc8e36073e0eda1fbf34f014bfe2c2873e21f3cabedd4b473fe44b6f3d59d31abf161aa13f91cc5abf350eda3fd5e4783e351cac3d44d101be519becbc74942a3f6d96c7bd693e243f92e0063f2184f9bed463aa3eea3b6d3df76b5dbfdfcf463f1bcdb23fc0e0de3edab72dbf554f5e3f93e7e33e4a78f8bd7a5b053e117bbb3ff31d3f3f4cde0bbeceeda8bf24bb3d3fc3693d3fb9eec43efa3012bfc98e9a3fd0f4d5bea407c6bf160beebf8092273f0820863f8204f6be2d82f23f1139c73ea9b413be54335c3e8076063f1b34b1bfd54f30bf4a1449bfaebb373f"), dtype=np.float32).reshape([1, 3, 32, 32])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
