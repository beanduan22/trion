#!/usr/bin/env python3
"""
Bug #0041: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['layout', 'resize_cubic_halfpixel'], ['layout', 'resize_linear_asymmetric'], ['broadcast', 'gelu_approx'], ['attention', 'matmul_4d_batch'], ['attention', 'matmul_4d_batch'], ['broadcast', 'mul_add_relu']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0041.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0041.onnx")
INPUT = np.frombuffer(bytes.fromhex("a3df73bea8b4c43f3dfe723fb8c4e9be7c308d3fa094eebe95fa1abfa636b5bf3dfa8c3fc29fdebf5fc7853f44b7963f5a47a03e031414bee86880bf32ece23fe9a3d33e53edf23efb9f26c0e60225bfd4ce94bfd2dd88bf688f113f10a39fbfccb900407b181c401389f1be89ca383f134da2be1d2db4bf1ec914bf5c56573f959c1fbfa91135bc36380cbf41b8b03f1837b03fb4920f3f93f3e93ddefc4f3e45021ac044554bbec289ef3ec305833f59cbcf3e5efea4bfd11b663e7bbeb93febb049bf92132dc009b6923d3533683f134d493ff3aacd3e1819e3bfc065acbf210c4f3b819482bf4d257d3e49e3b1bf1bbbe3bd57ec38be3e6c20bf321692bf9ab089bfe38987bf985115bf8890103f923a47bf24bb63bfb41c313f7daae6be149370bf0087dbbf1e0d70bf34942bc0ff00b4be722e05be8ecea43f8ff830bcbbc4bcbd11b639c033fcb1be5d8d85be3a0c21bf57305d3fd358473d87c273bf251158be0a20bf3f5ada7cbf8656323f02e162bf29b472bcdaef1abfcf5b9e3e2687383f4be5253fb895573f3229ee3edf10343fd9526b3f77e5a33fd05e8abfc41f8abe0f15493f44de0e404c5da43fb707aa3e66d7893d2759004087b225bfd3a70840a8ff1c3f688c3ac0e17c8abf935ef33c8777c6be576e823f4ccc59bf1f2df13ef186e2bf0ba8193fb2e06cbf698f08be9650dabf6024cd3f1e587a3f2014d0bf30a0573f0be58dbf06adc4bfbea14bbf73ca56bf35ccd43f3142afbe55b8233ff9de82bf2b5c51bf6341973d987faf3f2e57f9bd2ce7b8bfa3dbbe3ee63e78bf61d709bffcfed83e561b533ec74e5abebfd7e3bf9cfff63fa088c93f0afc213fb4a903bf9015433eb0793c3f647030bfb25e993d3410d0be9f90223d7de19b3f898e7cbfc05d3a3fd59496bfaa96a33f3594ca3eff302c3e629c5e3ffee3393f89718c3ee59c523d376f513e5a3a503f7cf4853f2e8848bf845ead3e8015afbffa1fb13eb90cad3f9b96ac3b0992823f977a04bf59ca14c043d3bfbf565566bf393c3ebfab5fd03fcb74a0bf441c2a3db3bea13e5bff5d3f8f0484bf40b71fbef143223fba96d03e7318083f399697bec9f46ebf74f3c63f07bfe6bf6cd7643e8aac98bf20a6e53fb7f2db3e2ff0d53f3299a73ef759c2bfd1acb23f8f47843e77c69d3ec4ba9cbfe03c97bf76d59cbe5f415f3fe4dba03f829eab3e674e923e8a4e913f768042bf170c02401e5ba3bfc3b5a2bf625774bf7e42833e494e8e3f6b8db3be486eecbf3b1b54be4b098fbf72b56cbfe3d9853e2a9688bef488253ee6a33abfa8a7a6be71dc473f713d04c0cfd403c02ad6103f526a59bf43ee9ebff4daabbe6cee38bdc1c0b7bf60ab1f40eab12dbf5e75043ff316fcbb5e10b8bee563773f5fe065bd1147873e27f722bf299e713e951665bfd59c32bfc878c4be41f09ebf1352543fce3244bf219f5fbe010310bfb708b33f4872aebffa304fbf7e5286bf4e74a6bd5527dc3f563abfbf778b1abf8e468f3e3c1dbabfea5f033fec27c93ebb47b13ca2deb63e8e909cbf4ad6be3cb287afbe868497bf0dcbbb3f4d8007bf3c615b3f44aa4d3f577655be6b8200bff5a71d3f6a9c0b4003ab23bf7be36dbf9729783f28e43a3ff61ad73f412c693e6bf9883f7fe4b83e6aa53c3f19694f3f2f8985be4f37b4bea845ae3f2646c53ea8a12fc07e1425bfdeae88be931b2740cf6c8cbe13d6ce3f356991bfdc63003f64e877bf64a89d3fa6f6083f0692803f2c5286bfd138babd50e5893f9df3babedb1e7fbf25e8b03e7ea00f3e62b6d23d522e0040ae8a18bfb79eb9bc3e7d83be1c000abe7acb733f7083a33e34fcea3e5394e1be4e5fac3e0a23c1be14b48fbfc5f3b53c5e6f273fce7db2be9f42873fda35253c280581be9621bd3cc5d9753e3372b6bf19fc08bf64fdb8bf5b1aeabe4a6815c034ad693f0b6c903f84ef8fbf52fcb9bf29545ebf21d8a1bdf74df23efd17b2be8e1de4bff1d35a3f13bdf1bffb19c33e3932c53e2bd73e3febf74cbf59908abf1848593fa65a85bf38df933f731b4dbfeb09923e19f5d7bfd01d003fd40efcbfcf7d1ebe92e7c9bea5f189bf29710cbfddd2c93dbd0c0f3f6d930b3ffdd2a0bebe79d73e8a92d0bd09acf73d7d05533f117e6a3fb894e3be93dc38bfbd4d37bfda17e33f2e029b3fdda618c01f3845bf3c98a9bf9af78abfe4dc6bbe69b77e3f6cc6a13e25246abf110fa23fcbdd913e338a99bf0f65b33f2895c7be71d760bf277f243fbda3f9bf3498c83fd5de2d3f66c9c1bf22fb71bea1bbb0bffda17f3ecd7165bfba2204bf5d181a3f0630493f3feb6a3ec7f1253e07399b3f74f7633fc4a18c3e688c90bfe61c73bef1c0a6bfdab0ae3e7ffda0beffc638bfa99d7abffa1a213f1c7981bf0ab7acbf5ec8abbd3dfa9f3f02bc63bd38c117bfa84f14bf539db2bfbe4145bfdd6588bf5c8595bfdacf91bf6c83e83e810a04c02679d13c16d47ebfc40835bf37fa173f1765bbbfd564423f1f9bb2bc26a1683f4dab0abe9877abbee39badbfb1200ebfd5629abe87434dbea00331bfcb070cbec98e113f17ba1b3fa94f15c0a8abc03f488d893f2659d53f8d12bfbfe2a1e9bd900163bf4d230fbf314201c0404f4ebfa433e3bf51e6313f8949b7bfb118dabed72fdcbf5b06f8be8661253eaf81323e8b59b73f7f0100c0198b47bf7f6d9c3fff4606bf8ab70bbf8209c7bf45230e3f4cb6113f7828ef3f71b732bf167e184000f54ebf2d959a3f8f416bbf95aa9f3fc64a6c3dfe82a73d657eb73d6151d63b1199debef8ff5d3dbfb8713f6613ac3eef46893f42167b3ff80ef23f7d996ebf9859343f5896a33fabe09b3e2150c4be0587084050a12a3f43af52bfdf307e3e59b031bffbeddb3f3276c03f685fe0bc1f09b2bf96417b3f66899fbf82a40fbfda913d3f5a523ac02ac0c63e92efa33e9c6502c085ed6dbe5747aa3bdfee473f862bd53ff3415bbe17fbf2be3b518bbf16304dbeee6111be6f69273fcc99a0be3653983e3b828d3f0e6513be41ff75bc07459abed98c543e42b1083f1cda213eaae2913f3c29d0bfc5195e408c3325bf906dffbeffe58fbf2bc7873f469344bfbba1c2bd5be79a3f1a8fcfbe1c9b26bf0a52f63e22c6e8bf037c88bfec77303fe091cebffc3a71bf4f3d00bf0df735bee2c9363f4bb92fbed60aa3bff9f604401073a93f66b616bec187963f420fdf3fbe9b543f31e5d1be5d9dcabefe6aa63f5fd6553f62871ac063c3123fe45d853fc3d4e83f66b2e53ecb0ee83ed3ed39bc47b771bec9dea73f4f5c7bbf3fab4b3f49adebbebd7ee5bedca031bf36acd6bee2bb823f315833bfa203c73f2c0c9abe4d97ecbe6f6ee0bea7a999bdd7f8883f0d12f73e064ccd3e874112c0d3eecd3e76bf93bf59f606bf4c9837bed9dba13f58d7cebe2b7b343fdc260bbf7835923ff5b218bf40822d3f08e6923fa5e92b3ef5796d3f85f9af3ee7318d3ced5747bfc341a2bfaeb98dbff519953f9c68bcbe5868103ff3d8c13f0197dc3ee71e3d3f400f7f3fae6ab5bf72844ebfd9113d3f66990bbf6809b3bf278731bf1a82b53f34bb173e01635bbe4011ccbdc7d69d3fca051bbfe6668cbdbd608a3f383e7e3e8f844dbf17c7bfbe0632babfa838983e6c0d8a3fbf891e3f31cb26bf1c725b3f20cacc3f983707c0c997eb3e2a6b80bf5d06493ef91e183e13468dbe802b51bf187efa3fc10fd73e92cea3bf3d5046bf6d4328c0271126bfa4d049bf9365f1be02f8e13ec566a23f16f6d63e045c4d3f4d64bfbf2de702c011b799bf9e681e3b3cbe1840e41dc3bebc9facbf7bb69e3e9bb349bf0de6d5be3d5e873ebeb0cebe8cce5f3d6075493e92fe653f711fe0bcc54ea9bf62231f3fab074f3f216ed4be96ac29bf6eec1c3f96b09fbf519323bf289a41be7210883c768e8abf802024bfe0e2c13e80b4dfbf908c6dbd0cab6e3feaea8cbfa64f8bbf8cbc823f07984dbf4ffdfa3f245dd1bf0c911e3f0bd8bebf45afe43b815f99bf6e88433e47bcde3fb45abcbf77210bbf9b00263f1ec68c3eb55b1b3f5ab5f83e90d4603fe10fa73f1c9c87bf77fcb1beae11edbee65056bff5f730bfc8410c3f5c7355bff8fcf0be78f8cfbed52b463dca98863f4077953f692b08c0bd38a4be87f395bda369073d522f753e6c1c73bf72fc1fbd540037bd8ad7ef3ed6f6093f862ca73f9e0594bf3728783f7ecb0abe9d9e193fe161153f2142f73e7f75ad3f3784d5be873b663f8d45913ec54a46bffa97853e7f9b2bbe97419fbc62f77a3f53e591bf87b9563fbac90a3f43de893e02940d4049ad53be59ba184065b782bf73ee8f3f0e93733d8edb903fe27916bf0fd007bdea86a4bf7c73e83edf34de3eb0e23fbc668d2d3fe2b6a1be1f4c9dbf463ea2bf5f71723fd4bc183f84c91f40f0b029bdcabc79bf45215cbffaf4243ea8803f3c07c9a8be2a8952be2a13063fcb400abfeaf0c33dd4174f3f3a020cbfa7e29c3fd7ea65bf1c3b46be2138503e60529fbf871f28c0f163fa3cb14791bfd43e1c3f7a9080bf9d7a2abfd2399a3f694d89bf74fca33fd19cc33e53c881bf459ff93fd05c97bf2f1d37bc3a46863e2feef03e37e9aabfc91cdfbf4a9abfbf4e12643ea52909c001b7a93ea150c5bf647a293e205171bffc854c3f9c73cfbe546116c045ae03bf14d3bebfa71952bf37c54e3eb08c963f46c706c0a93f393fe8620f3fbe1a0fc0a1f78b3f62f282bf9ebaebbfee4ee13f1af7acbf9574b93f36cda0bef4b984be4f24bdbf22d5a13f8f8c003f721143be25c0bb3f4a30e1beb395044047e7513f6fb3bcbf74bd6c3fc2f2413f1f24653f4d4a9fbe257997bf01eb8cbfcb56ea3e2ab81a3d2bde6b3f5fbdcfbe6ac886bd9ae1bebfd8c0b5bdf5e65abf34706abfa7be563d431aba3e038cacbf581ddebe853715bfbfecc93ffa559f3f6f42083f3bd889bf4f89a23d7eee1c40266f473fc31f44be136856bf31196dbe0ad3353ff974db3fda5cc33cb2dc7cbf9e472ebe1087da3e244cafbe5dcbe7be21e4b5bd4b93b73eb0e28cbfc2611040f69595bf9f0fccbff50075bf7417863ddf52523f21141dbe9c7f203fe57bee3eac8f3dbeaa5ec13e3f5d013f190f16be35a9bf3e809bedbf1dc83ebf1db2533f6c7a943fb7dc1a402a8cf13e9d840bc0bf510c3f52f2cfbf73682d3dd7ce8fbe8a4737bf1bb73ebf7dac9dbf1025dcbebae4013e70cf2e3f8c17d43ecad8ccbd3cce393feca39f3f8bb0003f9802113ff85c71be0873ab3e982f0ebf76398abec2febbbe2a270fbf7d69cdbf43bb4a3e9d69a73ff5b9efbf6bd9a7bff807813e56f91e3f9ce38abf0bd7843e8127e1bee4f2a13fc462f3be584cd53e3d2104bf21ff4bbf840502c023b73f3e2bb6053ec472e03ed09b8f3e9260abbe50866d3fc5d2213f57d85e3ef16d793e00e0a9be7585223fed6d45bee7a2c73f1228cd3f9346febee42174bf8ca58e3f57d0013f85681cbf3ac3ef3f92bf3ebf30fcd13faab2f23ea98ebdbf21f8873dda039c3f06b6cabd5268fdbfe4df533ff1cc9ebfe6b2ed3fb63276be0e1c083eb552c8bfae6dca3e3c52cbbf71a7393f836b6a3fc97e603edc7ce1bfba09753ef02f113f60eb063f9572b43fc03fd43e53d01040423c1ec0ab1497bf3200913e86530a409c8f084088de25bd0e2f22bf24b3ea3dc6e579bf94a084beb508553ec7bd923f35ae893ea63d9c3eaaaed23f9bab23bf01ff9c3e6f8fa9bfc8597a3e4b0d84bf33fef0be4ab47dbe064ee4bf98eb84bf92cd553efae6ac3f4232f03e8619b8bf3de4efbedbf6e93f47d6943f591c253fb0ded33f90f5503e249bea3fb818d2bf3cecbb3f0b5904c03e9b33bf656221bf8b4a41c03b1090bf13dd44becc662cbf0ba616bf5d4824bea852c73e6e93773e4f1e3bbf3bd9dd3e9181d6bea4c2073eea4607bfe0d8fc3e6e2c94bf57dee9bf9cd79a3faef7c2bfe7cf1ebfbb9bfd3efa45cf3d3d2d913febc6cb3cba45c6be55c6d3bedcf18b3fdde03cbfe85188bf9f8a9a3fcae46fbf26e2703e4c4997bffd158bbff2e53b3df715be3e7edf43bf28bde8bedd9dbf3fcffd08bf4b06803e80b7373eb9c5a03f221080bff5ccb43fe188aebfd071b53e1cb5ecbf919dff3e0953f1be75a6603f040dbabfd45183bd40fc9abfc3a6c2beabd583bf18b94e3e7dd6c33f3f64a5bf21bd29bfeabaaebedaaa7d3f9f0bfdbda396843f91e191bfdf86a9bf422ad0bcc78094bf1e5dd2bf3e12e7be37503e3ffa2200406d07963e5986df3ea291833ef702e8bf47c0cfbf480cf53fd2b4eebe51b5ef3da699cbbf5b00c4bfb032bebfbfc9623ff5d3923f6c010e40624688bf6a16a8bd6eb3b3be43bf383e23928dbe63a7dcbe65e28d3e4662f43eca846d3f56b0473f5591074053980cbf57a1363f1d0be6be4e1d303f140063bf48e0e03c77d99a3f8b7496be575156be17fcd73e02c5b0bf4366933fab6817bf9f38a43d9af8873fe24820bffb895f3f4c8f273ffb7988bf8f65d4bfb8f4a03e013ffabe192540be726aa63f2e62333f405c80be75c59dbea98adabee672f63e02e8973fba2c8ebfce9281bf8640ae3f9d31b0bd08f758bfb502c4bff85ac93ed4707bbf0a2fb23ff3fe1d3f402434c0c067623ea4bfc2be7339f2bd60eb12bff01a3bbf5b6b54bfffac33bf08c76c3fa95c183e08493fbea59bd5bd65b3a5bfc3c40c409c31f3bf22ab0d3f94f49dbf78d8a9beda2a99bc2d61933e9e8819bf38e75d3fdd7683bf01f516bf162b0e3e925d6e3fe2e80bc01ca1333f0b5f0cbf767c6dbfd85a9e3fd71067bec1930cc0eabd023e4999a83e01ec14408bdcd9bf932de2bd3015a3bf8efaa4be98a6d0bd7e90c43ffed909bfc0bfb6bd2cd204bf1119b03f3f1f56be10be2e3f2947f6beea9dcbbf8da305c05946c9bf9b9c1abf5619963e69cacb3eef63b53ec8c85a3ecc1980be977dc4bf6ed7bd3fd5d803bfeca1543e32ab43bdd8d541bf28f093bfd0ead2be5a1dd43ef4af963e9263f33faa8a84bf19d410bf039690bfd5ca0e3feb52813fe30aa73f17addfbf05e921bc9aa9b1bff4f04b3fc39ac23ec98a3abf8f31f9be0d80883f673bbdbe11ae6ebecbe0acbeab40ef3ede3be1be91e23d3f1fb7963ed36ed43f4d00bcbfb7c192bf3a44083f904c92bf5461453ea76d563f11461a3f4aa77e3c7fe9c53e575dc7bfe76704bfc8d754bf263b1ebf55d76e3fbeac0340f2c429bf8100f3bf3fb39f3f99ec29be28f56f3fbff7dabe5e1eb1be614c91be0aad10bf451b5abf6e65ad3fc5d50a3f58a031bfbee8b73e83e2f33e7ddc24bfc23992bfdf53f33ea408e23e876060bf62a0ce3fd8431c3f59fbb8bf4aa4e73ef641c7bf38c3b8bf4ccb04408b365fbf40d3abbd03b477bf94aaddbeb2ee023f3c4f8bbe05988a3f5fe21440d994643f06fae1bd4e5dfdbecdbb003e898152bf2f39c0be36593f3fb5a9683e2b02d13fbad1693f4111fcbe90338ebf32548a3fc20302be7a7df23ed7148ebef71c99be1eebb8bebf639fbf99ceedbe3b28fdbed938f0befcff01bf0e8aadbee02a48bffaa3cbbeece3ccbfc50024bf24e71ec09cc5e1bc9e3623be166f4e3ec254b93f5eae5d3e57348dbf18d998bf020a0140fcbe41bf5e879b3f839b82bebe577d3f243adfbefe18e1be3b46543ecdc40140fab9023f490f0ac04bf5713faf6be83e0dffbdbe59fc9c3f463ec1bef6108e3fe33105bf123195bfbebebabfacb604bff91ddd3f295382bf9b22433f9e5d6b3ff12b043f0bf93e3e380fac3f193defbff82235bf1da1da3ea54713c0c51a7dc01147efbee4a41abea2d63cbf7382a83ed29a333f394b20bf735e753fb65d5dbe7e8a6fbf8f96a63f3b859cbe4608de3fea9adc3f0bf1153e05685e3eebad08400477ca3f27f32f3d801f0f4007c43fbf58c4a2bff04feabddc59033fd3c6b13ef5e3023f16fe983f0ca9843f47910a3f417719beb28bab3f9ba893beffcb83bfaf11ecbdd36488be66aea13f28530dbfa87906bfbae43240d59285bf90404e3fdf321e3f2028343f8e291b3d2a79c33e37893e3d6fdda4bfd452293f1220fb3eead910bfd1ab5b3ed7e94fbd3d41a23d46dd7abe16f37c3fe6dfd8bfc7e1164058a5c3bed5c6003ff09cf83f1167853f2d8692bf14283e3f0bd481bf3b419bbfae24e3bf5299773ddb99a6bf82185b3e4deff0be1829cfbf35e5333e0ae5f23eeafc5d3fabfc1d3f172d90bfff7e383f110e6dbf9b097d3f38c6393fa527a1bff2545c3fbf537abfb7b0adbfb1db4c403cf6a63eece1713f166a5dbfbd14963f5c53dcbf2f9ce7bd85a403bec9c3cd3fb36b953f3a18c23f31cffe3e6a4228bf9dcdb83d7c845d3f16cc10be37c4a0beb2b74b3d3b0b5c3f1cdda1bf62101f3ed9838b3f7aed263e38111dbeb1378abedb303bbf0dc6cfbf28e6443f9da3dabf33327cbe1a7bd13faec936bf1978293e1ee9f4be88e3513e5da3a13e29e7c0be1c9bbd3f2df3d33f8f8d0040a62e06bff7f2eb3e49f5c9be24ccf93e1f3f903f8ea1983eb607febd8d86b5be52415f3f36d2054086c8d8bdb8d7ccbde56c34bfda44b83f36228dbe37d765bfff24b73e49b62fbf695535bf9d2db1be23b915bfd18dca3f9c6184bf692cd13e5147e4bef94aaa3e4f3b9f3d2ae628c075b291bd3b37d8be0db5b83f54d99cbf7264f0bd9f2d643f4260fe3eae75f43cb6849b3f69aba0bf83a7f53faccc25bf6c30cabe2abb80bf86119c3f26ed64bfe665933f2b12f53f0e51273fb40c97be923a983fbf8c073f53bdef3f965f073f3d7993bfd54f51bfaa4f03be48f5d7be09013dbf6cfc87bf0b44c8bdb83be5bf04af7abf63f848be94f83f3f2a98b13fc4f3873f1b21303fa0fbdabe95e3933f31078dbd5049f83f12d79e3fd6a79cbf7e525fbfb2e1efbd5d143e3f8b22fbbe9cccbebebd01b83efca4e73ead775e3fd951c83d7b70b43e3e1f76bfa7f9753f165005405b56533f0e22acbe5029c5be460c20bf938bc43fdcd06c3ecada853f5b679bbdb5023dbf626b063f4e4208bf9b0c183ff0a7b4be5ff6d33d6b9cddbf351e443fde8728be31540f3e098ef03e224fe6bf83dad1bf55df763e14b5353fe3eef93fc790cebfb86d0b4006cfb53fde50b3be265472bfa0b7ce3eb194323f30a5153feef0243e6c6d1ebf282c263f17bb9b3eba9886bf9f5643407130743e4f4e62bf72a521be5cf8943d1de5003f1be2b03d45e4c1bdaff12ebf3a015a3eff5e00bed8062abeb261c7be62495e3f4a7f03bd874f913fa6f9d0be767c05bef3e03cbfcadbbb3fafa9643e527cca3fd9bb6ebf52c9d63fa59045bebe266fbf5e547d3fc975df3e9ae1ae3ebcf780bfea540abfe13b9cbd5059883f0750b13f9a22fcbf5b518fbd7d32423f34e913bfc01c4f3fb2ce7f3f96ee71be824934c089b6a83e714d973fbe53283fdc00833f20e347bd9b9bb83fd322b73f269be63e4294553f9da0dbbe60f39fbde6eb523fb67711bf18f723bf009aee3f14b1803f6bed16be83218cbf618135bfac8e58bfe44868bf381482be92ae3ebf7c561540ed89083f1a36a3bf81dee7bec722bc3e61c6803e880e1cc092cf8b3e443ecbbf3f495b3f97d6e63e460766bf3ebc633fb1a3df3d266ab8bec0a469bf9aec5dbfff0b75be24af0e40d875a13c99d9a2beb1e4d6bd8bc4f33e4761ed3e7a037d3f32796f3e868c103f0f6be13e612c0ec0fff8e5bf4b0004bfc11f613e36681ac0d77d16bfbdd7533f2adfd0bf70a0623f79599b3f5bc6a13fb2210fbe46fa023f0d321b3f9705d6bd61e3ba3e4a7538bfd9679abff2040dbfe242083f838eccbf232986bf0a1113bf3e6283bfdb9d8b3bf9efb53d2e42a73f1492723fe030483e97aa843ffd76443f963e073f1da8f63eabb5b0bfd1c10dbf4cae193f263c5dbd7c708dbc251d8a3f40c2293ffd96403fd066c63f2a42dcbd163d0c3f40f520bf69bb38bf4e701c3fad47cd3ffa258a3f439780beef7cb5bec4d613be57e3813f5d9f4a3f5559ea3d530fa43f3430b83f6012443f584731becb5ad73dfc7c823e2c41ab3f99ab9fbf2f80573fa4f93340031386be047d4b3f878aa43f96d9b6bf1b07eb3f2bbb66bf9e75403f13bfc33ff2c1353f0525203f77cb9a3e735f9cbf6432bb3c9058db3fd67fe23f1ffb27be5760813eb48b93bf08d07d3f57c31bc0f684763fe790babf911edc3eb593633fe8f63f3fa5551f3f36e8a13f86546a3e536bce3f1584e63faba3a93fc671e83e0ae1af3d72f24c3f9e0002400f87e2becdf8913c20de47bf866550bea610cebe46e3cd3f0a0dbbbddc18d9beadc69cbe96f2ba3c2c1c3d3ffb96863f2cfb05402fa6dbbb9a9a763e691b89be72015dbe60a1f5bed34ec1beccb0083f32e8f73e057d95bf3e62cb3f2055733f4d8508bf9d0dbb3f027f643ffebfe3bf79bd39bda324e53ef0151cbfbc7d383ff869273f330a873f8ed6bf3ff91009c08c1e46bfe334703fd57037bf0c5388bfe2d29d3f0a6c80bfb24fbc3e95c46d3d23dd44bf6b3f8cbfb808b73efec0a13fa40ab5be4b5609bfb87e1fbf6cda943f06abe13f8a00913e07f543bf4d0000bfc2290a40f519583e9ecd1b406422b93fd4935f3edda9dbbf1a8a30bf42b31d3fbda4d13f90b967bf1f1bdb3ff4c129bfc6e915bfdea2fdbe6d29a0bfbb71a73d112ff0bfdccb5abfc90f54bf227273bfdac787be28ab643e3b72413f0870d0beea864e3d9a27953f7eb65d3f3ee1c63eb74b8b3f7890ec3f1a955c3eaf263d3e846e453ff672ad3f0e6af3be9143a7be5e82a13efbda6b3f5f6009bf3981423e6685abbfdd15bc3e46e33a3fd534a13d57cf4cbf4c7c52bfda9a5f3d8fc192bf6ed3173f662f18be449098bf20d2293fee1c733f9f05bb3c9dc93fbe58a1a7bf51d9013fa146a13fd507cbbe8f06eabe3d78ddbe5f09a03f28fa77bf779469bf63fd23bf14e1af3ff4da50bf9a9b7a3f5769b1bfc8fcc73e348a273d4a3bf239005dfabe744797be3e65a5bf114a9b3e1cf1afbd3ac7173f8a0bcbbe164a7bbffee7513f29a9a13e8d74763f9666bd3f72069a3f343d223efdf1403f2a2194bf11bd833f24c35f3ff05f0cbea66ca33ee2bc73bf6aedd6be6dac303fc65978bf42ef9a3fa6a4ea3f31f3c03f09150e404184753e91d2d03e20e77cbf4725973f5874473f7424c1bd1eacd63f0cec833c16fe04bf640e8fbeb69e8d3fd913db3d681f8ebf1d4a953ec4806fbf363c093fdd930b409eab94bef969d73e2b382ebe6958543f293d8e3fd724943fff93a43f048c7f3ff3c6de3fbd22893f55a316bee5710b404f69373f75b396bf46db9abf9c2f943e5099df3e289623bf3e2500c0c588893fcb6e87bf95fe0f40668099bf881d6bbfb282aa3f2e9fb13f909f8f3f75a1673fcabef43e1a25f43dcf0e47bf169c753f94615cbf3eab51bf4f971840d9e238bfad49babfe72da5bf0084353fb579c0bf65f1313f70b24a3f4ca455bffb9397bf87e557bd087f593e17fdb53ff8cefa3d6f1b65be090e12c050a4e33fb4ab3ebe381df1bd317d00bfa7aee83ee8bd4cbfb17ad53f6624553f5d11c13f57ed0ac065b4a03ec5b25a3f0ee9a13f6d3f474067981fbf650dbdbf13d4dd3f2d479c3f33d31b3f70040540f10ca8bd3042a3bee2a5533f715478befb05df3e423aeabd46f87a3e2edccb3f24b7093f7239ccbeab02283f68d9f9be4c549f3e2007c3bff425283f0ccc0ec0f93e90bfeaad1d3f36f038bf9b62233f4330b03f0322043e5fffdcbd2bb110bfaaf62c3f47f106be7c9589bdf27df83e892f14c0b07488bf3fe9353f9341ddbfaa5b1dbfb518c63c65baadbf1a202bbe0383a0be3a3b08c076f5713d496c983f0f0dd23f2733373fe0ed90bd28a5943f1f3961bf4296493f47196b3fdde73dbf7435a73f075792be84bb85bd7a57ce3dea983c3fa209d3bf3679fabec76a43bd7b4efbbfd24006406e589a3d622f253f302c123f0c62febe58595e3f8505f53eee43ddbf3538c1bff41e0e3f26f6b73fce598abf2751783ea32c9a3dcab00abfe42dd4bf70a216bf72a0603e8681113f9144f7bf3ae8ac3e4ae11bbf420aac3fd9d27fbfd78f613f2cf0f03f1126e03ec86e6dbece38003f88da0f3e2b605a3fd7819bbf8c71c73f2667cd3e32e18c3f660bcfbf045dc73da5efb3bfb1a3313f530cb93f5219043ee1878abeef264c3f1035aebe39aeb0bd348f9bbf72370d3f9a2c0b3f11a5a9bfa0731ebf14cdf8befaa637bf34c4cb3e31019f3e5e8b103f3772363f0227873fd2a0403fcfd88c3f5fe3403ee531443ff950a8bdd0f22abd53bf0bbf556eb23e24bd9fbf27cda8be1e1c88bf816d83bff8b244bf16dab43e0af0a83e5a7576bf8e8477bffa4e543fa7d9013ff1f3c8beb42d4cbe371cdf3dc03d5e3f9ac297be20b5cbbfda136ebf931bb8be7b999d3f721a9e3e1abefc3e620095bf0a33d5bdaac395bead01f53ca432373fe1d5193e726427beada59abf5dab903f883852bf4aac33bf7e4816c09b8e21bfe58395bc2c637f3e76438f3e020606bf9cc64ebf0f9f1d3fb208a2bfca9d483f4611a83d644227bfd2560f40a72eb83c9098f3bfb33c963ecab5093f19b4eebeb4c0953e7a84633f4f5e3fbffd94a0be40380b409e2013bee159c33e004996bf4e5aadbed0445a3e5a0365bfb1b6dbbfd1aa9fbf55dbc9bf122ef4bf0d1f903f55301bbfe6c40b3ec8fc92be567b7abf672c093e0a73d6bc2493b3beb4597bbfab7d0cc0d64dcd3e8073f3bf6ee4603f61f5fc3f681b573f769c123f61bce2bf30bca0bf2abdbfbfdfa1ffbed5d9a4bf3267e5bf7082f3bf45b414bfcdbf02c04d87823e5862c83fe885a53f0f4fa63f3868273ece4a9fbe8258f23e21b5f3bea523ec3f68dbd03fdf17783f520ebabd8007a03e34e7443e8f56273f809a623f9f02423e01e7bcbf65c292bfe0aa453fec6ade3ec0cc10be099c473e2d17603f4cb78a3eeeaf4cbe578a9a3fba3779bfb5dd8e3f1b87eabe8783a43f18ba9fbf4e0642bf7c20b93f3ca1da3f28b76d3fa73da13e81a48ebee618113e71b680bf19e37e3f94a8dc3f338c0d3f897c8ebfd1509abec608b4bf3c3a1dbf85c7cbbfffc75640ec6857bf4f5f6c3fc4a2bdbf2fd6bb3e17409ebf3e8991be3cab8dbe0742a43d87a54d3f69d4d53e5e9b013fcf2726bf77a043bf85716bbe7fd2013f30dc7f3c431b03c0be48e33fc78accbe49f9053fb4cbb8bea776673f5f48443f3b35683e3fd580bfd997de3ef5e678bf263dd13f1116bfbe4fa9283d9849ce3f718a893f795f80beafa90cbf48b6233f7d47cfbd038caabf5aee6a3f28479c3ffe5b2d406663ecbe1d19f4be7693493fb5c5673f2567c0bdb2d2cebed2ecc5be03c1173f3cd503bee0d3fa3ec6f564bfc38150bc7f2399bf706c81bfda3e77bf86fc073f12ed82bf8013cf3f5ea02b3f42555a3d7723f53dab6c683f90f2c93f29649dbfe403903e78fc61bf5dcbc1bfbe1ef5be826e563e873b95bf87899dbf38493e3f302cd13e4ad1303ed2c2bb3f47e5bb3e1d27803ff8ba7f3f6f801c3fa250753ef4ff873f2febc43e2884873c57fe02c0e44ce6bff76f0040616d59bf3d09083fa488eb3fd76c553ee5969c3ebdfcc03c490ab9becac1c23f543c25c0570632bf8ef635c07b1a47bf766fef3e7b0cb93e101d95bd04a0dd3f0bc13abe9d5fb83ecb7e193fe9d97cbef58f02bf23de1fbeed95373fa6b3febf970e3cbf6293a53fd0bb7dbf62e6663fc0e22f3eb7d2873e7125debdccf722bf16ddb23e8a5fc8bfbba3e5bea511e2bf4d6f993ff5d2963d96d90d3ffa5eacbed9cf593f9dfee53e1882963f0c939bbff23090be8aefef3e4df577bf9e50bcbe40cc9fbf11ae4f3f87fd573d569a2fbfc2b5a63e265dbb3fa25a7fbf184e02bf5612d53f4b289c3d7d91083f0a2c823fc402d03f0874a73e2031e33fda49044018818a3eb33311bf63e46e3dd70443bd2c70f33e155286bee20fb33e8294fcbed01a28bfdcf5f93e6e45b8bf4d7998bd5923ba3e4d4d9abf6098943eadfdc03ef419113f2f16323f02dd233ebbffd6be8d2cec3ef27042bd93341e3fb023363fad69d8be92b09bbf5a2ff8be7110523fca6046bf5a378d3e531eb03e2b3c15bd9645383e063d723ef29c1a3fd4b607c06e945b3f72a1b7bf1694dcbf78790f3f3d8deebe486aa53e2f10293f5fc2843e2301e93e2dd4b9bf5637a2bf9e6aa2bdb78a5dbe2094bcbf90c98fbf176f413fa31916bfc11a03bf0a9fa53f347a1740240419402e90b03fd222283fadc218bfaa1944bf1c7c853f8a96e3bd00ec3abf0ae869bf64bbad3e29f95e3fd3223d3febaa8b3f912985bd7f24eebe325f9f3fb117b9be893645be3a5d98bfeac9003f79f0b7bfa27f883eedd21f403ebc2f3fa721bd3ed1437abfd107d83f6d2621bfd9b28a3ee576dd3f69fd7bbedf2390bf3082e03ea6392d3f04b0bd3e32571dbfdf49393fc6af63bf6ecdcdbbce9f1f3d8a5721bfbff9a03eb91d3e3dbe2679bffaa5b1be07f327bff5590cc0817768bfd826843e5916313f872bcc3edbe3a73f396a243d087f07bfb1b1d8bdca42a13fb3e9fb3e73765cbf4da665be2841f93f1e9b00bfc967c43f715642bf0a04e5bd9e7b443fa52bff3fdcd2a83e23a35cbff25ddcbe0f551cbf649952bfb0213abf9c08f5bfb5f46ebddb0a86bf34008cbf7769503d093d03400e1d8abf4b607abf00ee78bfc74ebd3e06852fbe5b14cc3f815f0b3d8d2768be22353a3f019f8f3f867125bf382d28bf36841b3f8353243f14c3e23ca695b3be7a2b8ebb2c96493f9c72193f277d7dbf2989133f3dad2f3f3727173fc028793faddb8abf4a94913e748815bf0b8e693fa3e7b13ee4cd703fb22910bf50c05a3e3f03b5bf03590c403e48153f152029c0907e9cbe2ed6bd3fcd453ebe148935bfb54c903f19c208bff9881a3f534fb8bf215d92bfc58343bd942d44bfff4dc93fbd28f53e0fe81b3f31b1553c4b7d81bf2947353f4f72b63f5f808cbd3f8b13bf7a0cf1be6329ab3ff145c4bfa37aef3f42a5af3e9fda09bf72a80ec036d3d9be3dea1dc0522fbdbee0fc283eb6c3413f416542bf34a04fbe2eed1abfc387123fdaf5c03f83c4c2bffc96a33ef37bedbf2ccd41bf564a973e7d04263e369e0ebed97d6fbf4019e6bf7543afbeac54b33ed511a9bf4682a63da184ab3e7bc278bfbadd053e966c0bbf35dba03f92757abe8c9b293f846692bf86ec563f8110f33d5b65e6bf0d0f20bf902b83bfa170b63efee0b0bf8dfe54beba29203f3d792ebf7d5c2abe2041dd3e5a372c3e21cc69bf85096dbe01e105400a945d3fab51d03f8cec333d5adf993f467cc03f65aaa13f7f51e83d30bf4cc06175053f2092b93e8e5b0f3de99d63bec49acabf51bb5dbfc4c0fabe78a9763f067d24befb4fdebd234fb2bf70a4c1be73576e3f6321043e2b9cd03fb15fecbe06fb10bf2a7812c08a1fbf3d3e37d33fcbe023bf73503c3f1ce6d2be0e2a0740f1f2c73e6ab120be115ed4bebc520fbfe57603c0890b463f5d8fc4bff794753f98c9b9be7e56543fe77e6e3f76cb733f913c833e61c9d3bea557e1bfeb8621bfce70493e3bb12ebe56e7b93ea00779bf3d2480bfb2fd55bfebe3763f37f8653ffa36453ff29371be1b20523e76f42d3f39da83be37b6d13b042d6b3fe05a13bebca6903e85b6cfbfc81d47bf72a9873fee8ab63f99f318bf3b24d3bec966d9bf677e01bfc6fb3dbf00563abf8bdbd93eaec30040798c67bde3a0e43f7b2d35bfd25bafbf23dc1c3e7781ec3e51c2a9bf43027fbfed7ee03c41dfa33fb161b93dc08016c0456b00bf1ef3bfbfdd26813f558e01be1b08473f20c4393f81b3083f7f98453f56dfdd3d9258dcbffe62bd3fed2a534009f419bf302b93bfc3cbee3e83f1a2bea01b8c3fe1f2f53f4a18edbf853257be6a9158bf665b763f0792963f934c0ebe0c7b16c03ea6c8be8230153e11d2f33e5cf8523f4f9608c02084223ffef11cbe835652bf9d12533f280cb43f348248bfa89fe2bf389e87bef464a7bcdfcaf8bec3d318bfff18fbbe69a397bff95c37beee2c66bbc4310f3ea977143f449d40be3a058b3f0ea6343eb4cc8cbffd7775bf4512f9bffdbdd33ec33a25c06446fabf4b20793e356ab43d6525c3bfd2188fbf26ff5f3f56d4a7be83b8853f33cec3bf38948d3fc51c393f0db8b8be42ac7bbf7c9f7cbfd7509f3eff48a63ff8902ebf801972beed74073f381325be07bfb6be0b3820bf422857bc4e36743eeec17dbee08dc1bf369ef23f5716583e5a2002c06ed6b1be1ea553bf9dc9a43cd871d1be11ffc33efebdf13f9561b7bf8e82993ee0bf91bfce24813f79eed8beddac81bf8a691a3f53d4a0bea3dbf6be9dc5273c5cfd463d8fed2c3fc077213f9907d23bb45911c07158b73e5e5500bf9086f0bf3e6e813e6c7305bf4233c43e009634be4875b2bf764abdbfdc662dbf2779653f843fcc3f37df9fbe2ec0a3bf7713753f9c4d93bf40b613bf89524fbeb430d6bfa9d9673f5bed163e611d9c3e28b5e8bdbdba813ffec4b7be069da53ff18921bd0b62bd3fefebd6be5898e2bde8c9b63fc9b6cb3f3d5331bf54873abfadcea0bf28c743bf5c929c3f0524b1bf86938dbf05c0af3e43b9663ec6d4dbbe5063373f10f0b23e09c98bbf0e8677be17b3623e972b623e32e567bf86f6643d1030aebfd65fccbf1582d33ff131fabe714c303f849504c0f840a2bfce4b0e3fc207333e0f33b43f782676bfb2fb9cbfebab643c4e6b0d3fb97b02bf2836aebe5243b4bc96a77c3fe13b7cbf90a3993fbcb620bfe9203b3fced91b40db065abfe8d8293fdf8a2ebed672283f81d637bec505b9bd36f1453e6b443bbe2302d43e0ad5cbbec2ac7b3f485ae6bf469229bff776a43e862189bfb3e8cfbf52a1e6be0f61dbbe314206bfb45a3a3ff462eebfd6b7b0bf565ca03f043ec0bdbd49803e77b7bf3d7161333f5a348c3f7bf3153d62922dbf20a7f93f63b6d4bf7d38fe3de7f312c0f838a3bddf2c503ede25d43e5a218a3f2c7a88be6f4c833f84cbc63fe181903f409d8c3f9550ac3ff512aabd20f38cbfb98a993f881eb8bfa1dae03f574a03beb17fa53efe3da33f629b6cbe2e3c573fd25f9dbe0f64053ef7172bbc3d0ab03f001f863f9f4590bfd47d4e3f9d1aafbeb281393e0e7413beb73f5ebec6aa07bf828b673fd371d1bfd6edc7be7930063f064e2abf9eff9dbf8a6974be00f049bfdc1d633e193b5a3ee6f01ebf"), dtype=np.float32).reshape([1, 3, 32, 32])


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
