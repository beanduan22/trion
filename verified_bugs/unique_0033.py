#!/usr/bin/env python3
"""
Bug #0033: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['constant', 'identity_chain_two'], ['fusion', 'conv_tanh_head'], ['layout', 'slice_pad_concat'], ['broadcast', 'floor_unary'], ['broadcast', 'ceil_unary'], ['attention', 'matmul_4d_batch']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0033.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0033.onnx")
INPUT = np.frombuffer(bytes.fromhex("a9f21dbe62b8b5bf0345a2bfd4cd6c3fac248ebf67d9203eaa4d6f3f5a16df3f89dbfbbe5fa481bf22010240a16b2abfda61823e9800813ef9ac883fc873833ed68ac9bfc23a06be326e70be9eaeae3e1e23c2bf0fe497be51ceb23f248791bec83acf3f3d53a3bf93c4e5be09c1debea09dc0beec26c23ef5b9b0bfd034003d76c01ebf05f774be08da3cbf76b1bebf99e413bfe354773f4805a23ec4eb023f0cc821beedb8c03f0554ccbe2831fe3dab3b1dbf29f382bf3eacf6be9843d6beff2710c0c8b1c6befff6e0bd9e081c3f3c40323f04232b3f72414e3f4a4b823fca3a90bfa1aa053f30a21abffe3896be15781ebfa813443e0777ff3d3aa88e3f7d0911bf918743bf319fd5be84a242408de4643f577f49bfdf0832bffe8cd7bfab853d3ea712b83e9c50b4be244fb2bfacb625bfc1c4553d766c513fc7ab843ea89b19be0a70093f082ef1bf298acb3f7c4b513d3b029fbd310e6dbeb3f1ccbf863b6ebf736ad33e222033bf6e81b8bef8d12bbf09d88a3e981ca0bfae27d7bfed8e543eb17c10bff8dce83f1df5633f020e383e9f24b03fa542a03d92465dbf074d9a3d4156d4beaa4514bf118a4c3ffaa1b1bf153934bf1a0ac13c158894bfa53bb93cc614f3be96d662bde352f7bdce0309bfa252ee3ce36e05bf309a213f5096debe523e3d3f667ad4bfd75babbf7460e93ea9cdeabf10dc57bf8b4cafbe9f6cae3f4c9e01c00a4e3abfe140fc3dc234b43fc00ac4bf37be20beec900040567eb6bffaa6d83f89813abf9ae4ee3eea48b53d84b27dbfce1bdfbe40c52c3f65bedd3d1227843ff48a8c3f143336bf3a1115bfb3cb10403c5f773f3f177d3ef31001bf3562813ff2d371bf9c6b90bf4baea5bfe1123b3f341f283fe93246bfcb6135c053adf7be66ca49bf14e182be5e648b3e3d6d47bf5722d43e59ec953f9dcca13e63f04f3f6f4552bfff3a03be82c68bbf66aed8be5ba8943f9df18bbc5175c9bfe1e9953ff6ddf43f9182a9be3b6e87be1544ea3fd53770bec6951e3f95335cbf48ab9b3f433c463de89e2f3d05c79c3f9c02963f9703d5be805641bdb8c6063faa038a3e3510653f255365be864a36be1f5f16bfb7c932bf98ec84bf8774a23edc329ebf4bcb7abfc60c143f4d1744bff9ceb33ec308513f248a2b3e3811663e49ef1dc00578f23e2ac660be6e5273be41ab7dbfef30873e8d5774bf4e1fa4bfeb2634bf40668abfb443acbd6a76a6bf11f11fbf004031bffdd13f3fa935c33f844b01bf494484bf9521a23f61f6603fba29f2be0d25ecbd2229453f776e2e3e87938f3f08b66c3f68da4c3f7c923d3bc025cf3e9fb8c6bf461adfbf7876edbd149ecdbe46fd563fa4dee9bdbd3f863f51b5393e0d4a623e449cc23e851a45bf5897b3befa7b2abe0ff030be268b96bf7ba345bf25d1e3bf56b9273da3f3f9be75dc79be7012a03f0ed7d5be0f6ae8bfeee9b43d44f3f6be37f57a3f001103408469c2bd2a1f2f3d064e213fdcfdccbf308c32bf03957bbe83c18abffff8c43ef0baacbe1fd1cbbe3a75d5bd98be0f3fad3df53efaabb9bfcec1813eb0fa17beb0ee553fd6b8303fd613303fb7135a3f52a03b3fd13bbabf8bb38a3f7466833f9160154013e8473fa1b046bf608ea9bfab8e613f1b9060bf90b425bfebe8463f8a5a8d3fe24da8bff439483f6c53e7bd5661173e9054e1bf3fc18fbece24c73f699676bf01bc973cf1e7e63f6d5d943fba598d3fe4ffe9be59cf50bc42b7e8bf2eec8e3f268275beea55b7bf22db95beabbcebbf42a8113f9a3cc1bf4058c1bf584380bf019a3cbf7f238a3ff41bd93e3ca6273f9e2c363fcbc0773e2a2da63f46f38d3d868c30bfc05d5ebe972ac63cf6cce2bf665e573ef8c0123eede6113f8a9051bf034c9cbeaa140dbf33f5f23ef8e24ebf33379d3f6066cf3e510be6bf3179c73dfecf8b3fb778b3bfdb54a53fdfc7a6be230410bfd2f4103f1a857a3f02761a409932e3bec66dc93f772e973fc2dfe8bdda45ca3d5329ee3f7a6c04bf8098ca3ad19ab1bf3c7cc33fe4a45d3f3c392bbfcab3f1bedb3e883f290d973f9fc1603fabb3f13ff2d9263fa7d399bffe09583f147e8a3fb213853fcb12b7bf04845b3f2a95b3be49d4e7bddfa0bfbe681bd83f6628713f65ec143ea242b23f72c9c1bf7c6e553f21a37e3f0f40dd3e0797bbbdc10ab5bed8d2633e4a74533ebcdabe3ec15acabf8cc73ebdcb99c73f58a9ca3e7dd1513e660dc93fc03cb5bfd29b123d8d5e803f37bdd1be0a5c7c3fbc656abfe4a21c3e956920c0a52627bf05fc34be9cc106402e6308be694bc5be3031373c4519a53e914ea0bebca39e3fb2e0203fc38caf3f1dc7023fc6d5543d3c70353f3faf9d3e5a822c3faccac2bef3890ec04cade2bfec0437bfce2dcfbfc529853fc53d70bf700cf63ec0191c404e15a9bf3c3781bf6a0edebf1cde2d409006e13ef568613fbc2b61bdfa91ffbd140bfabe7c1339bf90418bbf828a60be9951ca3febbbcb3fbfbcfbbfbcec173ef8856f3f7141253e3c3a513ed1ae193f51f3d6bf949db9bf6dce2b3fd9d00fbf40f83b3f8d3cce3e57b71fbfefde54bfc4062dbe8c41353feeda52bf1020623eae1b98be6861ed3f305800bf00b588bf2d74c5bff301bf3f5d94843fdca2403fa96b5c3ffc9b6d3f457fd93eeb0018bfab2648bf403dad3e992841bfee4382bf2698203f31aa0abff30f273ee0d019bec27c17bf0c9a92bfcece8cbf5c69a2bd025c16bf860ef0bfaceeb2bfd98b4d401ccfa83fb07e0bbf1f27d63f0022fdbdfe7de23f80b0a93f6829753df64a8a3e1aab853efba828bea183f2be28b32dbe1362ad3e3573f2bea3196f3f49d2eabf0ff209bfb60614bd289936bf96b4c9bfd5ad18bfcdabaebec3cb9a3e2e75ef3da481883f2057c03f12bb3c3ebfe8b63e7305083fc47f7abed48019bf833fe2be9d3d8abeea14e33fb36037bffa95f13e84faf43f7fb582bd4047193f85c9783f66082dbfb914533f4b4930bed6459d3fce33d33f79e9e33e135355bf1e2bf23d55a0a93f6f75903f1250983f8f7eeb3dc1d9a23eec50f5bc132756bf388c0e3fb5ad903e4c7c753f23c1aabf24625fbfa22a993f0992e13f008ef1bf965bb13d2bcaa0bf1c0a7ebfa9cf9bbe880051bfa08714c02fc1613f15b586bf736927bfbabadf3f8e606c3fc80bce3ef3917b3fe01c9ebf66972d3e443d8cbf3609ea3e6f80553d549c433f36f5a63f300caabf92f254bf59b36bbc9910e83f051815bf28d1a03e8e7075bf29b16e3e8830fabf0e1726bea47b173ff433f53efd2018bfcb04febdbf5b33bf037cc3be7b300cbf29270bc0880df3befc83c63e6e748a3f3161363ffcbc3bbfb094e83e51d8353faaaaa63f753ef93e4d95bb3f9eee90be717e0dbef89ea8bed0e6973f31e9de3ee807683f02f8c7befa7aa13fcfcd33bfcc0d10bf904bb93de961cebfa96a1e3f57f6183e9eef5f3fd8535a3dba44473c4e5d1d40431f1d3f2e63adbd5b70503f909ef7be1855853d38a18ebf037193bf3ddfc4bf4529883f998514bf1dc2dabfd52391bfebb99f3f887f9a3f9ec1973ffd24b83f2edf04bfc3d813bf5a4272bfb17eb93f5364a23f1fbf7d3fd25450bf1496a43f4c85973d3d2dc83e6b4170bfa6de903b9b4f9cbec56169bf1e34aa3b60d300bfeba48fbf140c13bf2e1a023ed73b183f1becc5becf08c1bf697bc13ed2aea0bf4b1a8ebf7fb6ee3f78e406bf870637bfe780a53fc2b07ebfd004353f66ea32bf09c5b63e3021f33fd7b399bf2bf99e3f4e10183f22bc973e38a9bd3fc8a8a0bf11bc9a3f0726773f7036273f176181bf8d15c2bf168ab8bf6e02063f4808aa3e59b9b43f6620b53f1fc4d6be7835ce3c7030023fa5e49a3e4591ba3f6595c6bc60c9f2bf699c15bf35c6743ee1690b40cf287abfffb08a3fda42fc3f22529ebf88c7853f37f6333fefa1b23ee9130f3fa170124012c712c0275696bdeef9ce3e7fbfbbbd80bbc5bfd044cc3f84ebdc3f0388a53f1f6b0a40d9c9b5bf76b28abdecdb28bf6ff39e3f8109e73fa7c209bfeb9992beaf40ebbf694b943f1ef9073f0807bebe4e9a053f5b79ccbe270cefbf580b27be0cc2133f7a062fbe47e79abf3c45a3bec6dba0bee671adbf9f4d833f942468be2611873e978bf13d838e3c3fd03d0c3e708deb3ede3fb3bfa4dc4e40c438813e2d6ec13e46366fbf129b0dbf4a5dc93ff0fe35bfa59c733fd04a73bf5d7e8c3fb151eebe3ad4223f3af7cf3ff881983f32cb44bd5c9d4cbf90e87c3ee38947be207c313f09c5af3facc0d1bed40bff3ffe99b5bfa41ca1bec97ad8be4b8accbe3dd30abf3e2cbebfa135ad3f3782883fa020ee3fe112d3bf21dc493ff70d043ea60881bfa76b2b3e32399fbf3f97fd3e13bb6e3f55234fbfd3914fbfc5210540f3c5bc3f58bf053c31ed62bfb87174bf66dda7bef788a23fadd4f9bf96a6903c1870833de3838dbd7fc36dbff4c2ea3f726a14bf9b7a723f5dff153ecb71963f453eddbd2eaf5c3fa12d04406420acbf0b6d11bf634e893ff253a2bf924a13c09e6c56bfcb85bdbe5c3386bfd8ee3dbf90a1d73f46e880bec5772d3f6bd9eb3fd1abc4bed839333e1da97e3f06f28cbf7af8b73ed98938be0e5ea5bfc399afbd32008c3e1672e43d3c9beabea3d050bf88e61bbfa1a08c3fa78e07bed1eb84bfab18e03f6bbd0fbff349543e30c382bf9442573fec4cce3f9dacc33dc7ea24bf90f8a6bf5edb943d8835c93f50b6983da089e03f4483344057f352bf77b4aa3ff359d43d83fec8bd6e8e023f899e803eb1a62bc0ee2e73bf24483ebfadae13c01521b93e0a29d93e980c09bf42c0f4bd1b303740d17f2ebe1ac1ce3f27c4e8beb5e91ebf45a663be393efa3fdb7c3e3f5b198bbe9b4b95bf5ecc473fa1475bbfb840bc3f9b8b11408638a8bfed1edd3eaeb2a6bce499403eec1aa43f012674be0ecd2fbfb0279a3e762e59bf2fe6cabff3504ebf5d771d3ff7fec93fadfa9ebf5b70863f8786ffbedb63ad3f9cc77cbfeda623bf2649bbbfb9aa31bf89c00fbfb2702dbfcbe6b3bd908ba0bf1de2053e0e28a8beec12383faf6c2e3ff954f23fcc7892bf415332bf80ebb83f2536413ebd68a73e312c763eecc477be900e943e50963ebf94b326bf254222bfdda976bf4941debe6213e53e2046813f4be8983f8c5bc8beeeede6beff906b3f2bb096be0f6e383d12b78c3e8f29f7beb6c4b7bc338da1bc2f56d73bbdcf19bf736d94bf7aa9363e8f59893e311238bf8291e3bf7f4bc5bfe9e6c93e788bb03e4017413fd510b4bf3158acbf424017be3af4a23f419cc0bf6e1668bef637fbbe2c82053f9f0598bffce2c5bfff9a57bfc059e83dcd9797be7d4b243ecb735e3fde6a7abff3240f3ecc64f0bfb16915bff18198bfeef266bfe6e218bf3f52cabf22919dbfd64fbabf11bc2bbf75c1593e26b5df3f620bb13eb11699bfbc9205bfa597a23fc8914c3f933fbebfd12c933f478f743d63b469bfdff0f6be818b44bfd68c963d8c79853f7510a4bf4bf3f3bea2a67ebeee68dc3f65326ebfdac0553de8def5bf1fa90d3fa734db3e512fec3ec648cf3d964fdcbe6fdb4e3fa38b703f1c96b7bf00498a3f5498bc3eab6605bf473534c0a866683eddbfdcbf41081bbfcfc4223f78375e3f60bdc9bf1774d43f8ea8943e4f7f2c3d6bd7d83fbb42b23e8ad184bfb5a25940d033d5bd52a9a9bf078667bf0532973fcde65bbf55f7aa3f52420ebf70ed8dbe7ed5ab3eb2c3843f655e9abe66fb6a3f9489323f4c6f17bf00fecd3e7ccf44bfe4cbdfbdbc3b02c026561cbf98dde83dbe86e33fd955dbbf0fa4573fee85253f9e20053f360d91bdbb4a8cbe73199fbfea5f7e3f698ffe3fc144b73e83a063bf4f34053f32f803be0f17653f99b5ac3e9e82ddbf44988c3e3a0a203f3b6d59bf53090c404b8a8e3f7e221d3fe034833fc08bd73f52b64a3f4dbee63de0d805c0e42417be8fa8c4be6b43193fc49312c0d8c7e93f50369ebe4264173f34b624be1c3cb93f1153333fb7661dbf34f23bbfb6aa143fb21f27bedc0263bcd91b713f18f5d1bec81d16c06fc087bf1ade1cbe8614eebd61007dbff1e896be4d7a8f3f84433ebf894b383d2feb04bf0f09ffbedcaf15be27dcd1bee79e1ac04e4f93bc4d17abbf01bbfd3f26790f40ec87973f7beec13fa4b6113fd661bf3e18d1a5bf7644f23ef60b46bf038b963e019e84bd96c854bf4e5ff33e889ff73fa158e83d20925d3f5bd69cbf5e5cddbe34c9263ee7ca85be1b3dddbe6dac68bf8e21a83efd8d15bf73c8b73fa2e4fbbf28abcabe0238323ebda87b3f6a8de13fa3a8a03eb642b43e52edb8bf16a8ab3fa20c5ebf2daaeb3fb7da65be8588a0bf05fc3b3f1dcb0240d94f78bfd1bfe63e56b78dbfc49f3fbef5fddc3fdb42b83f5281293fc549eabeabd3b8bfd254293f638aaebe15e22fbff806b7bf8548673f8108babe3ae929c012ece33e71049dbffb0a223fc0758b3f034df43fb35b6e3f3203debf205d063fd3f15fbf64bab13f7ea7553f9d1b8ebf0675e83ff683103f1d35d2be7b95d03ed1ffd6bd3a35903f6210e5bd4079f8befbc7bd3e7ac30dbe099236befd6c0c3f072d8b3fcb33ecbeb643233ffc743abff322d43e2db120c0e74b25bdd0d9c93e20f0e23b85d6e73f98a6024091800cbf70f2febf8c3ae0bdf434953f0e8b69be5e172cbf7c5bba3f3fbd76bfc657d0bd963a62bf6368f73f26f8bb3e4031913fcff9163fbd04e23eceab9a3dbc4d3f3ed9d120bc4c3740bf1436bcbe55d8e7bc849aaebe9802ea3fc087143e8d25053ff3e1fdbe8654813f0839563e5409823e160b0fbf88b5693f331e9c3f548bb23fbd184b3fac60903f2be9c8be1b5261be0e5aab3e1eecbfbef5264d3f9fac0abe1c6592beb88795bdd1c186bfc5e033befc6007bf780366bf502a8d3f5bee153e1ac60f3ef412afbf68362dbf6c7372bfe63ca93ead4c63be3a72323ef942a33f98b654bfeeb4adbe0dc9fb3d14a3e5bd41edbdbff262803fbe02b63e28537fbeb9de113ffc478e3d74ee8d3f45c4823ed0670ac08e9bb23f917a7a3f02b612bf91c533bf3197cbbf2e15943faee6503dd2bcd53ee64e2fbf5892d03f1eccf33ed6d5e6bfcb7571bf88f4263f69eaa13ef9ca4dbf77080b3ff630a6bf696717bef88419bf9c9124bcd667e6beacb8bfbed387813f4dee683f18f1de3e13c5293fa114a3bfbbf00cbffc3a85be82df60bfeea92d405ebb85bfa5e04a3f205553bfbbdbaf3fdf2bedbfe52e233ef27d19bf13c99cbf103306be92ab863eef11acbe48c1133f553a64bec3b5643d509e10bef997a73f8e6a843f3892143f88bf55c0acd2b53e4831d53f565e993f8e207cbf81273c3feb10f23edb71a6be067388bf8df25dbf25d2bf3e17b33c3eaa0fa3bff24c043f9c544ebfbd4246bfdf3483bf98157d3faf49863dcec81cbe048c333fa3bb953f79eb6f3e6fb5c53e3565683fd3bba4bf2faa33bf3df17cbe7c29323ec23ebb3ee50704bf3ff077bf4bdef13e001f313f9a9c48bf0751273e5bbb81bf165afd3e2cc7313f002780bf9b67c0bff800d93f06e9ed3e2fb8c7bef5e0013e71598abfa84590be6bc150bee2be9b3f977d96bf48bf2b3fbd80b5bf0fcc153fc0c8e3bef9ea16be06ddc3bf187686bfb83a083c2e8f9abf69e8b53e3bc4243f065ffb3ffe43d7bffc3168bf62a38a3ee88eb8be5c6e3dbe65db743f767b90bf55a689bc7712013fcbf7c43eb07a293ef4e708c0c63f143d9879223f45a9d73f3029563fb67c2ebf3595723f0786d43f7b2a6cbf293d56bfcb32a4be03bb8cbf002c863e42b43b3fcfe2903eb51f2dbfe54a8f3f76247d3f1b38febc9651813e0e5ea63f633727bf1fefa0beb38d943f3b278bbffbbec5bf7ba80e4020b54c3e88da833f8b0e953f59921cbff4a6d33e2dd56a3eba1991be7a5c9a3f8711b0bf3a0113bf09d84bbee7263a3fc0155f3f57ce53bf4f0fae3f38aaa8bee7358abd933badbc112d81bf91ea1e3f79a2a8bd081cb13e073de0be8b1189bf43d055bfd5bc21bf306b883f3124113f911b61bf914e1bbe4c27df3d7c1fb3bec8083ebff99f69bfc2a6a2bf551a3b3f108cc9be9d76143e7b9a4fbfca39aabfe3978f3f971b183f9b1a7dbf3162123fd0dcc9be0c56dfbfec43e73e11b39a3f9b91073f9c6148be298cdd3eb265acbe4a8aaabe36284d3fd7ffcc3e4e5acd3e5d61c6be0c579b3ee096a5bfe1bd34be3e6c5dbf19f4773f2c9dd0be4f51073f2d375fbf60ad513ec2d5fbbd8a448e3f0466b5bd3a9e0c3f7d4c22bfff7d11bdaa81ce3f49dcc03e30b68fbf5a07c43fb3400a3f04d3843f7ce7adbb4464af3d98d155c013c9debecb50813feae1903f5837313f8548ee3f69aa3dbf627881bf1e72d53c021c94bf0715853e6f8d913ea23882bf586867bf01730cbfe3a4d13f95447c3e93a383bf716fb03f8a1e123eccb0f23cf0cf863e8fb219be9658f7beaca78b3c47f44bbfa4fa74bf7c34bebe4f2abf3fe0ff0dbf8484b83dc6dff7beaa54473f554a2f3fe37a15bf31b0a53f96610b3e29c06bbf9d29353f3f60893f67061e3f9bb8d83e349385bf5534843ff9581f3fefb9803e3c4ad8be24c2d23f40f89ebf73731cbff65656bf4f15033f16d79dbeff3934bf5bc4ffbe9818f9bf698ff53fd9c78fbeb58c273fda1b40be6d691bbfcc5dc4bfe82397bf6fab713f5c291c3d553a343e35bfef3e3079603ff6a6bebe2ebb19beac1d7e3fc8e32fbf4c240ebfe283113f94460bbf64803ac05e42bd3f79572d3e1e6183be6c93dbbe87b16c3f5665183fbeeddc3d512120bf8caaadbedb5c85bf8efb4cbfd659bbba35ff813fc443edbf0bebacbe61793dbda62eef3f08e258bf2a7f21bf388503bfca363f3f1d76e23e1193bebf090789be1711f13e77eb663fa196a1bfdcbef63e6e7894beb4aa133f06ac9cbd95a8f13f1435833e06dad1bd5a6008bfa2d94f3ed657393f8491a8beafc921bf4d9466bfe3fee73f9794b63f7c788dbf8ae31c405d9ed03f02cb033e134fe7be621ee43ed12c93bfa9895abf71608bbfa997d63f4f1805c0d43f2a3e5fe5d1bf7f274c3fcc6fb43fd14271bf13a0a93f68f6c03e354dde3eb249b3be9ac3a13d888ad63fadaba3bf66ddbe3f92f7b4bf0230d83f7a66923e680bfabfdf56a0bd6c0a6ebe1045f2be2dc8cfbe0d95c93df8dfe03e8bdc25bf45e7373e6575f93dc0d7993e81ece6bd38ac14bea6dfe9bf47521d3e8e89f03e46118fbf72cf8ebfdc0108c0f5689e3e77143f3fb769dabfb3875fbfae8d4abf1538013fb771bd3f664efa3e13c4f4be7b318a3e895cc13f343d003ed832f5bed5fd9bbca933163f29a5d5be24ad50bf7cd24e3eaa41e53f7810fa3ea093e0be9d10543e6c7b78bf4b21c53eaf309b3da7b78d3fd7ceb9bf3223bd3f6f76c1bf5e16f3be7a5708bf002f943e87da88bef99a423ec878f0bf71e5eebe247d05bff2e86ebd717c56bdfc430cbf16f6813f3eff063d89e4b8bf196000407204893fcc54d8be2d2c133fc9e6243f865b303d34aac2bf14e5b1bf51f92fbec719123fcd8ec1be48fa194041c85fbfdeabb8bf5088fabed1f323c00145d9bec2e949bfa93838bf5865d23ec95e813f665b093fe64d03c0e8db5fbf80cc24401ca133bf0144373fee01963f7754b1be76ceedbefa6826beb7ccf53cd821773dcbb9773f66c32640ad3410407eae8b3eada3603e36ec0dbf3a57bfbfc6498f3fc258ecbf133586beacd434bf42c819bee51ee2bc8ba399bf226d863fb7260040c2d66ebf1b15acbc8f8c0c3fef721f3f8a30bcbede01c0bf1fe60ebf82b3a2be72b208bfbd14bbbfa2b3e43fafb2323f8a51ffbe371085bf01a82b3f2cc1853f943f523fab80023fdaad313fe7333a3fb9c9ae3f62646e3ff8a5b6bd8082593f1ca387bf50ea37bfd1a2c1be100b9f3ff502d53e76c6e23ebf5f0a4075fe973f3a69293f48a81abfd865543f94cb39be28db7f3fdc48ddbed4c167bf7746d3bfdaa6d9befb000d3eb0288bbc43d7e93fcbcd1bc05532d63ff1db15bee9dab9bec7897e3f6a06583f9715b5bea17f85be75864a3fa7d8d13f1f60d2be7a30363ffabe63bf91a5f03e8a48024080b3343fda14acbf112cc0befe4203be8658eebfad1a8cbe3a8dc1bfb558a73feb3f92bf630344bed36d9f3f5634c5bf063fc8bee4d9cebb3a0daf3eca0358bf109ce1bf4cf18f3f75357dbee501a23fbabc4e3f5c2e34bfa3fc1ebee71b003ea5bb843fd58b953f265e15c0641c1b3f723019beaba4453dcc71833fc52a3b3e936aa0bf70fb8e3eb857aabd6a6b053e868ae13f55c2f13ed2d50ebfb8008a3e9b1172bfcd6598bec75a1b3f936d74bfc38c59bfd5411dc04d794a3f0721213f87cb0e3e6293ebbfa73c373f092ea2bed26021be0f3bfebfe68a03bf11d54c3e4c4a653b5874cdbf28b5a03ddcc1ba3d47f5b6bec1cd0fbf5bab64be531b0940ba545cbfc57c623d13c7d43fec757b3f27e8e6be22d1b5bf724ddc3c9c796c3d2ef3eb3f3a50af3f7713e9be17d178be9bd71ebfca1a603f456701bf8a2f3dc076e20bc07d936a3ddeccd6beab25f0bebab106bfa62474bfd179803f8244223e2a604c3fb013763f1de3643e49ccdb3fc106dabedced59bd72d8403f8b20423f8c2a39bf770b33408e07cf3eb0b1033d072290be42d7173e77c294bf4aee2c3f59bb06c0663d4f3fd54d043e80d3023f49071ebf67cfb6bf618f0f406fd8e13f888f903f4967cd3f33ceb4be6075213f2d185ebd5a3d863fa15e5ebe5d7202bf568261bec436debe156edc3e411834bfec84383dfef7bd3e2aa4cabee3ee683d2dbccebf4cb79ebfcd70b3bfd445873fbdbbc93fcd01e83fecea29bffc1c693e6b0f6b3ed727b73ed880d4be8192ebbe74090e40cdc1f0bed9fdd5bfac40e2bde7d5e03f2b3003406f8d86bf2075123d48e9a83fffe9bdbf9ad547bfc7d609bf8e3e1e3f5a1d82bfdfdd1f3f26e8b7be54be693f51de453f98653ebf4ba4a93e24535cbed7e0b23eee410b3dc599903fb778553f1d7d2ebed9c3113f480f61bef44778bfa720c4bf2c2c7f3e3c8307bff9b69c3e2f5932bfe33eb23ec89a8c3fe589b4bf4cebb63fa618313f17cb2c3f12d1aabfba7ac2bf45de77bfc73297bf07740bbf6ab2ee3e1b7bd0bf9ab12abfeb0a7dbf419030bfb9be323f90682e3e00e2a13ed1a20dc05c820e3ffe259dbfc2c8a0bf8007d53efc6dec3e68d88e3fda7d9bbed156f0be331d2ebfecef4fbf77459a3e9c8d783f2aa347bfb4bc8c3fe6007c3ff441d2bf07140040ab7f2d401fde74bf7a6f423f3e14783ee691563d9b19d73e4da339bfef0a56bd48d4763f276590beb288b3bf0ed9883e8c43ee3f4bce283f7f6d99be6cd4d33e9a4c4fbf3e976d3e2bbfb33f951a27bd954c95bedf09833f795a00405528fdbe049422bffc4516bdb8a6f7be1b5c493ea9039ebfa355f83f0609f63f7eb40e3f900c9c3f6bfbbfbfd11681bfdc0e0a3e168e12c098fd6dbfb7aa68bf24ae3740bc7ba8be6447743e537632bf0fd5d83e1c53cbbfd98a743fe46020bf7f7bffbf4acc0fbf35615c3f196a123f7d44b8bf39c55abf589d65be895d3d3d4d9d9f3e170fce3e29add23e2aa8a7bec705e1be6577973e8dc2e73f7a3df63fee09a73c5ac2fd3ee83d07405a2063bfae8600c01e6f1b40160a4d3f1b45873e0eab3fbfda42b83ec9561140aa375c3f13213fbf0c3edebf897b9abf1ed9b83e1c0a18c0d388febf1fe2b83eb7e92ebf0872133fcd4df1be484f16bf699c48be5ef2e23ff147b13f76f93cbf1108cebf9ae6c1bfd50d7d3ee245943d94291dc0ce06ca3f12bab4be1bee7e3e10aeb3be805adfbe15bd9abe51060e3ed50fb73e3656a23f4711a73ebc1b53bf2601acbf90c0d7be769725bc1cd3823d1949ab3fbde2bc3e05ceacbf1ea1693f520f6abfee7a08407f4e2c4043d38b3fa8c5fb3f4f82a1bf5cf4a93f9882d6bf05add8bfb209373fa8b62a3f9ee0de3fd4c49f3fc559c9be9ca3e03ec5aacfbf596125be7d6529400a10873e04f79e3fdfb283be75a0e23e61641ebfa62c2fbec160c7bef8d43e403d758bbec2e3fa3ff4656a3fb82bb43fc6ddd43edc55d83e9ede843eb6f0e23e8a565cbf56f5c1bebcd58fbdb5362abfe886afbe72286fbf5bbf393d39f58dbf0b4623bfee1c77bd09a2973fd85d153fb022793aca893ebfcfa9a5be049f90bee49c933f6eb16f3f5bbc91bfb79c3ebf52f9b2be1961183f8bce2d3ee5e80c3f513517bf17a422400270f5bc2fc5863f55b5d3be46ee09bfeddd00c047fc59bf93b399be810668bf51664ebd008b89bf5265e2bf068406bf72ba063f2e87943f9269873b744f64be3a8b8abfa0b41e3fc06da93dd555b13f0be16d3e333b1b3e6e861d3e37ea0e3fa50664bf5b75413fb8800bc05b387f3eb14713406f85dbbfe82ecbbe746941bf652512bff8f6cf3da15d2fbf450bec3e0b960b3f4c075dbf86cd333f08f116be987462bf6185823fedb132bf56a249be1f9609402f8d4c3f43d0933e963837be1e92353f56471c3f5c1cb6bebe96af3f5508553f360f943fd9c22cbd9cc6bdbe8c7462bf02dd6d3fd5996d3fac23793fc56dff3e83a2d93fd94c59be125ec2bd3c049e3fc5d790be1fec81bf5ed4053f2eccdb3e0aa5063e6b8980bf7837843f90648e3fc94e503ed0024c3e3e499a3e07cf5a3fc4840dbf0b7a203e8bfa7abd04e7d13f0d47afbf72ad36bf205c55bfdd8a513f628b373ef0061d3f37a1d7bfbe841abffda65b3e8eeb9dbc69bf11bf1c5786beb1399dbef699a63fcb649abfbb5361bf5a6bb93f547b72bdf3fd02bf995e41bf671f7a3f6a8b5e3f840a313fd1790ebe409613bf01e2b53e00df06bd7746a2bda6f4c0bedcb8c8bd0b88003ef8aa9a3f4b44fcbd5c75edbf5f13ac3fda6319bf8aedecbee3cc8dbf79cb0b405ed41abf110ab5be115cbdbd117b833f4587debf3138cbbfe3089e3f06b1bc3f10afa83def4301400be3b63fa2ca1f3e40226fbf5ad276bfec8693bf45031e3f24be3b3ef88717bf3465f3bd3327f93f56873d3e5b8c43bfee04aa3d5bcc9bbe8b3840bf7af5333b719a923eef5418c0aba2773fe677193e1349443d9a51f93eb2fba73f9f5a07bfeba39dbf44240ec017c868be75efa6bbaa4a383ffd4e053f331746bf06028dbfbb1f813f57470b3fb6f8aebe43c2ae3ff8753fbfcb00eabf594d983f8c28773dcb25823f22c257bf35bfaabe1b180f3ec19e87bff719053fc469c1bf3fd0fcbf517a56be52889bbfcbe497bfd8c14c3fcdadb5be5bd66abc391e6f3eda7b4cbf599ea93ffa3e9dbf1ba3f33ec402603f00d60bbf25e134bf75bc783f929ccdbf5b65043e3a1b334056da5bbff12fc73e5d2ad13e2b0fd8bef3a3a7bfe2912fbfe18b17c02d169abf55bb5bbf86477dbde43cab3fbffcb33fc4b69f3f7dacb3beeb74f3bfb06d853e091715be50815fbfefa1db3e08debcbf3e220e40d41ea13f64db64bf037582bf11d3a93eebfde1be4625313f482e34bebdc2013f3d2c893f8ce9eabf52f1353ec06eed3eb44c02be3d51a03f8d30043f5df3353f1d8b9e3b4b6b463f71f5c5bfda4aefbc8678463eab76333f25edaabebc3e14c08ba003bf26a2c4be0e4e1f3ffc4018bdb44310bf0e0833bf8853a7bc5323813f355e833fd559593fecadf13cd5df673fa70d1abf50771e40cacf013f8a6f21bf1537843f111cef3e13d10d3e30f695bbdcf31a40998c803f61383cbfd18b083fac30323e865b9a3e4dda23bf852c1c3ffb2dd43f7cb095bf8bde0f406bd617bea8164dbfbed8f1bf2819babf8de4fc3f411c00bfbb12ac3e81e8823e9c66693fb19ba4be7ebe2c3f248d003e8cd600bf1e7f9e3f34c0dbbf646ad8bff28e8dbe107a783f8a6f24bf17a1a0bdd6fc8abffdc920be4eaa3f3fe30a35bf59fd5cbf6a5a2d3f31e4363e86dd393eafa294bf92bdc2bf75843ebe59a1653ff4a3cbbef4fb2cbfd0238c3f5c9b1040d726293ff2c47d3e321935be7063973f3239064086eccb3fa7b7563e84d1933fa3e774bf0b448f3e33fd423feff490bf7a367dbd59d0653fc34bfd3d6a5d513eaf9d3bbf4a71263fc96cc33fec1329be5a9d7b3e015fb6bfdfac07bfbe849fbe40531b3e9d2fb1be2cb193beec9a443f135dd53f48c4463e7de95e3e7f599f3e22e070bf5951c83f50510cbfdea4bdbe2bd7323e06a753bf92221abff01618bedf773ebf83c980bfd21e7bbfdce4823edf646f3f745e23bec7fa11bf3ae0e5be156824bf702d183f02bc9dbf8adfb83ffe6bc23f0dd4b4beec541a3f967fc7bce4dfa43da063a63f7f5c163f2831473c74996abda5cd1ebf6a48633f94a53e3f8b063ebd79748dbe54baaabef34fc2be890ac9bf39c5653f322d9ebff4d7d3be7983e23f6bdff1bec70fd6bf9fc6653ffc2c843ff7572d3fef7f723fc79cb8bfb51db73fa55b113fe23501bfc3630dbe8094b7be6a7e3dbf2d5a9b3f7dd404bf9b3fddbee027133fbd4a6e3ef5b9b53f0d3639bed2c89ebf1b26713ef331d93e1fc2b93f623000c0ebddacbe83acec3ee21fcf3eea87113ee093113f805252bfaf6f2fbe538cb0bed0bb1e3f99a5ac3f88c9f23dc732d23eedcb04bf55a80c3fd8ea303f815bbe3fd82c363fc9e15abff414e63f4b40b9bf8653853f2d740fbe48951dbf272fa2be18301240d8768fbffb59c33fa91069bf0607ca3da7a5453f03a9103ed884f6bfe0b56fbf429e25bf18f34d3fd245c63eb0f1b4bfb0e3ce3fcc0adfbeadc8df3dc7f0c8beab97f13f65ca05c0645cc8bf8afc5cbe41d412409aaabfbf68bb763de20d0a401ebb48be137f2abfc1513c3fc5e0b33f691a40bfeed8913e575934bfa1b580bf8ab89b3f6e4e21bdf10fe3bf548d953bc0f2223f43ea04be01f46b3f1ba4773fa7328e3f5ceed8bdcf9c0d3fdf59d13fc736e33f5f1519c03d56743fb886833bc58d05c02021093e085c42bf8654e9be4c915c3e9878983ed60bc0bed3bbaf3e1a0a583f97b49abf9cfdacbec52cdbbeb91578bf01e9113f47c8193fc1b5b43f529393be5c308bbdeccafc3e7f0a31bec07a553f7e4e17bd9388e13f1189ba3ece7380bf0f8e933f2ce89bbec21e01bfb6fe3a3ff93b373edf18fb3f185046bdf19c69be95fb8fbf96c865bdb74d22bfe54a4bbfa82e983f985cd33f7c4e4f3f88dae6bf2137083e00dc8f3f4a21673eff46473ecee831bee0ee7a3fe4b97cbfc35ba03ec21b29bfc65b85bf39bb35bf477ad3bf433770bf4c9ac83fe1980bc0a96143bf319302c0683c023f5a538a3fab9c613e6e4cf5be17aaabbe0ac7a0bf4c57ad3efecfc4bfc286a03f3bd5163fe181aa3e04f34ebf4eeed83f3ad9c43edc2fcc3e3fccd33fc8f3383ee39b3cbe1d52cebde0d6353ce6d686bfa272503f74790f3fc2b685be6df5cfbf35ae1c3f3c67ad3f0febbfbe81d77e3f03f83b3c17a307c0a8a8c1bfd794dd3ea381b03fa1b299bf5b35313e6daeaebf37a3523e172bb03f364e12bf7a84ef3f1425fe3f78d1fc3eae0915bed25eee3eeef878bfa9c6eb3f4f09103f57b4cdbffc57083f3c3d6e3ff7d8853e30010b3d7ca2f3be6c2762bc6001673e9406fb3e6dd5fe3e632698bd80d7b03f3e20b03f2175663f62afc73e03f0123f1a518fbf1ffb523ee03c18bf4025703f870badbf1788bb3d21bf33be5091aa3f7a78073ff493b5be10d873bfe9a62dbfcba2823eaba6d8bef70789bf01eb93bdb11612bfd039253fa4ce5b3e03d82ebeed654ebf0b4dd03fec9d5cbff9b001c01cf9783d0b170a3f0956373e2605193f10eb4d3d75940cbfd1cee5be54a0ea3fa141893ec21fb7be067f923f6d0a8ebd7a1818c071f1253f83360a40f6f4a9bf0ef5283f81a930bf11bb2f3d52fe28be30069ebf3a80cdbd2a8481bffc9e913f6cb50a3fe0ba0cbf6123e73f275e903faa7efcbfe6ee7abf5609623fe41919bf9a2832bfa78791bd709a0cbf6cd9713d7511423f902602bf35b665bf6d55893e044b9f3e020f5e3fd5f08fbf1c25e4bd2ff9bb3ea8dd77beb3b03fbfa954653f83e113c05cd2313fa167da3fe0ee03c00738e03e93a45abfd06050bf7abd403fb245973ece29093fac9c94becdd0db3f43579dbef97a77be5fc600c0acda1140e11f30bfb7044b3fb6a2cabd1891973d4bfab33f00d6babf40796a3fb551a7bedc958d3f981c713eba5c67bf9c1ea83e2c46173f02608c3fa64d043f2f786cbf3f1aaf3f5701a3bdd2c0a73fd41177bff6354bbfa960903f108189be4c53d5bec6323fbf7ea4aebf131ea33fba1277bf9793a5beff9530bda88974be4f76fb3eed71bdbfd4189f3f06124b3e09182f3f158e02bec8ecaa3eb061b53ef4ed993b80d646398dfc0940390cb53ef3db86bd02dd803f0803853cc1eee3bd967ea23e77e973be5d9ed2be2cd3b23f9b2cc3bef00ba6bdacd5bcbe6f33d43c1d670e3fbcaac03ee348c0bee00279bf4258b5be9e47c3beb576853e66f586be97bac8bce45318bfa12803400c023cbffe2d2f3f7b32a6bffb04963f00e1c13e8923bbbff41526bf105cd8bec0b68f3f1257383e8becc63e3c0c54be0640b2bf231218bfb338e4beca01563c47861cbf3eb3773c8aae29c06f00013f61dde53d2649a9bf4a0b94bd585c13bfa08a8cbeec619c3fa5eeb3be1b2d303ee05d163f5775efbf45ceccbe669d993f0116893f1265edbe6bc2333f70e9b9bf2f669c3f8baa48bf283209bf0145843f9c2a2abb2b1c7d3f60e0b1be5112fe3e01893ebff377b33fef2da5be36414e3f48362dbffff2493fb74328bfedfe2ac0f8f69c3f2519733e8260a7bf1e528abeb406bbbad74b89bf546c35bfa433653d0dc00540d8d3cc3e256d12400ef2c13e4a3380bb4ba2c73f8b7edb3ffd52cabfe81ad3bb219a4cbffac0c7bd9580b53f7dfea93f646109c09f5a063eb37fcfbf8f9d80be5b6ffa3fddfb96beb13043be352683bfa8d6b8bf0321c7be546650bd3843773fce281d3db1d627bea4a3dd3d2af9f4bf7aec26bf9920c5bf0f6a93bfc35010bd08168a3e048070bfaf93cebe4b35de3fa0e2dcbcc60a253ffaec4a3fa7cc9dbe0c426a3fdc61c6befaaa94bed63b42bba24c043e8bd9bc3ed41cb6be4f49893fcc5fcf3f3eab583f5e6c03c0ff8914bf48daf03ca662933f5bcaf3bf2b4ec33f3f966c3fd96382bf5bc20c3c42c83abd26a8283f3c09923ef55973beafbcabbd03e4263f5fff79bf49f8883f9cabb83e"), dtype=np.float32).reshape([1, 3, 32, 32])


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
