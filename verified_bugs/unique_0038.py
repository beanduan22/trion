#!/usr/bin/env python3
"""
Bug #0038: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['layout', 'reshape_layernorm_reshape'], ['constant', 'transpose_identity_perm'], ['normalization', 'layernorm_axis1'], ['layout', 'resize_cubic_halfpixel'], ['normalization', 'layernorm_temperature'], ['normalization', 'layernorm_dropout_identity']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0038.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0038.onnx")
INPUT = np.frombuffer(bytes.fromhex("45445bbe0f3bdc3f4f666b3fdb4b793eb0fc623f3cb3263f5f28d9bfb773ccbe90331ebfe70fb63e4b41ca3fbb54e13ea2c896befe6f373fadeb70be4a6a9e3e3b50b93fc5485abfadf388bd41a5863ea2febdbf00cc6f3f93a9393d38e20b3ff3c98d3f7d41cd3e80079a3fdf9c064056f1333f8d7fcb3fadcf36bf4946c63e738c223f7f11c0bffae3ff3e09be86bf5d578abfe8f91a3c8af397bd684b1abe18321f3f33bf80bf16ed88bf5a85e33e883027bf23cd8c3f674a52bfc30d64bf9eb89b3f9d5fd13f5fe8d6be1678b13ff265c83edc96f6beb1c837bd0ac7a83f09f6243fa93612400ae0a43f552cb3bed0d0c0be7d5833bfa38b2dbf9dbc8c3f54692c3dbea499bf9e0fe03ee137ffbe0d8da33f9c1a2a3eec9cb03ef32cd8bf1da656be3ae6373c1356093e58abac3daa469b3b0bf109bf0e2bad3dce3ef5bdecd2883f6a1c263f6023df3e356c633f58af833fe87006bf076c0b4090a55bbe2f328cbf16bed9bf43c3d33daba3333f9efd3c3f9f9090bf30ccff3d398bc73ee7a9bc3eebfb523d4730833fb3c7e1bf74bc1e3e8539853d8447cb3f75a98c3f407ebc3ed5ef073eb327703fb0a51ebfcdc05d3ff2dc343f09bfdfbe2052f8bffcf59e3f3666623f5f59c1bf8ccbc03db6388ebfa6d4f43f6221d1bfd508c4beb9c4b93d051939bea8a6c13fcb06023f755bc83e7953213f67e3453f20bf86bd535c513f5dabaabe12222240332b823fd70f23c0bbc1393eee1b273f44fee53f2208abbe000a61bd4e17a23ed71bd73f2fa7cdbc4f8037bf3a54383faa2a32bf0d239e3e447af53ef27554bfa9a149bfa2fc73bdf013b03b681dd3bf449e11c01419003f12bcc6bd0738ecbea4f0163fd65cf5bf1c81843fb99814c06460aabf70c0af3b7b61f6be5d9a993e400c8cbec64fad3e5d6a82be19a3873e5946ea3e3ded62bf9695a9bff976733ff08c0fbf1a50ae3f8ae961bddba6f8bedcf97bbfa82027bf40b3ee3c9f0da0be427bfb3e68b497be780120bff39e73bf244c28bf158e6fbe78223f3e1aaaa53eb6cba83e23a3253fed442abf177504bdcf589abfae2bfd3fd75f88be4ae01fbf09ba2b3fe012873fc0edcc3f5e4ac53f54587fbf41acd9bf96f3dc3fa86ea9bd021d1bbfd0789f3fdec49abfeb9d6fbf4547ae3eba5e393fe422c33f224ebd3ead800b3ee567a2bf1db9fd3fa494e9bf8e15bdbe9096ee3ddcb1fbbe575fe93f1a47a6be13ff2f3f0cbd7a3f7115e13ef0acc03ff8b6163f11b7ffbe81949c3fdc77f3be27de28bf3b8f25bf3c42f9bd999f9b3fc2e983bfc623b23de91b6fbf634388bdb58e513fc9d0203f8d0da7be7cd11ebf959b343feedf17bf199abdbf6856adbedb71833ffef33abf90658640ac66973f80d0033f44bd3fbeb89273be0b41debee823a9bfd5df64bf2df40ebf7cf834bf081ea93f09cf79bf364e12bf28c5e43cc40e8bbff3ccfdbe2621f13e9b6cb33f00ac1f3f8e52aabe6038eebed5007abfb32101be6c81b0bf244d34be8f34e03d60bae03e91e2b93db8ec21be22faa0be78a8bfbfe5ead93f17e5913f7510423f5e817e3de1a8f7bec5b6793f6e0b073ee4e23ebf6ba9db3f44f5b8bf316a8b3e44a909be83e51ac0bbc5e93dd81728c02331493ffbe4823e0845b23fbb10b6bebd3f0d3f77cd84becd3208bfb83f50bcd04ec7bf977e78bfdf21e9bfcbaef63e2ed8183f3740153f0d71f83e409a1b3f50e7b03e0c3107bfc6cd6fbf37f32ebf29888ebf2330bebe05d102405738093e14d391bfc251023fd380293c0212c03e56330fbfea49d6bf96ce583e777effbf8d0874bf845cb53e10cdf4bf03d8a3bd4ae0df3ea76fcebe591ad53e6ebe2f3e7b7b4d3f0974dc3e0ec6213f8f1bef3e3f60593e401ee73e1be16e3f131a8ebf8928823e276185bf213d4f3fb97b31bf7d22a13f57b5c7bf5aa6b63eb5ec743f9691acbfd5033cbf4fbcf6be2e625f3f6a1e303ffe93acbfc603c8bfc62f81be245e69bf9ef7293fda1d813e6ffcbc3f847a593fe014413f3293d5be15c0d1bf6a510d3fc6c4913e2182b93b610c5d3d7ca7e83fdbebb33fb111523f6b120140b0f5cbbf0a50eabe544be73f3a0852bf15fbf0bef8c1443fe4f073bfe92a3e3fd207a03f9106a53e4c9eaa3e0889483e17acb1bf69b89c3fe51087bf8d5c103fe7dc93bda47f9fbf20b8c53e0fe2b4be5e69003f4c0357bec02857bd5315b6be6823d3bed0d6a03d6eee7fbf30cda23d97cf54bec21adf3f909d7fbc0dcb673ffc0d84bf6333bbbe96a9833fddababbfb554f0beb8b522be9997bdbe61214ebf804f87bf3b951dbf5d43cfbe21e0d23e75773f3f2a231b3dd0ac94bf8dbbb73fe733bbbff42b363fe96b24bf4712bd3e1e4eb53f8c73283fe43d2cbf762e9c3d5fb1f6beaa3dffbfc820dcbf4e3405bf3f54413f1c31b93f80b9f4be4e5c893e298d0f3f22224a3e08b8c33c103a623eeab574be0ae1c1bf2afc6fbf5233083f6ed657be6b11b73edeebed3eaae57c3e2c64abbc9ce9083f358470bf1f560f3fcadd713f7f792d3fc925afbd90f7babf2b9a613f907fbe3fd30704c03a160e3e1bfd42be81e1c53f507b6c3e61ca813f9c72d13d03cb4dbf9eb2b23f85d9fdbd2474a83dcbe038bfc4d862bfd8e3bb3d0f14933d1423d33e4d67a8bf230033c0993073bfecc8fb3e8f11a03e3c4438bf44121e3f83a818bf19b1ac3e1bf2c13fb30d22c06aea9c3e37e272bfadfb00c07b9cce3f6736ffbeb2f0fabe1fe9a2bfab8cb9bf75840b3eb62c733e6123eabd39cc75bfcfed62bf2d5b88bf27ca97bf583eb33d46227abe0fc4a33edbb6e43f353bfb3ee3253dbed8600bbe292836bfd84f9e3e6cdfd53897c140bd717b85bc1012853ff5ceeebe5f878c3f6a78cfbec733053e79900bc095323b3e563d8dbf4e6b91bf9054c63e29d4ecbed70a693f48a29b3fb4928b3f313797bf7926f53f04a3f03e188528bc3ff8ac3f820540bfec2602bccf1140bef243f03fcf3d89bd4e04833d5c8f3c3fb4ee93beb7cb5b3fde8099bf840ac0bf113bc7be284e833f827c943e17bb533f1f49bdbfe5942e3f34b104bf0ce5063fceee2fbffbc278bd303e97be0efcf13e6113f0be7da3d8bea41c1f3fc570743e257c573f133d2c3d9a20a23f9857abbfc3ed02be535b483ea0cf02c03ff48abe433f1abf4de6f43ed4136dbf7e0abc3efcd40a3d8200873e73200dbe00d54d3ed9df5c3f30dacd3fa8bcdb3f961f80bf90068a3ef3789f3eda86673f3dcee2bed88533bf32a416be7011463f024d943f7176ec3cb313b7bff4791cbf6213773fff28ca3e0c2ead3e4bd0ebbf4399653f582163bebde9043f44749d3e0861993fb03db83e438c2d3f3dc22f3f5fea76be27e5fabe47e99f3eb5d3163f0e3d7fbec61090bfa90e883fccd0143f753e2dbf7e508d3f1ec41bbca682cb3fda04413f7f8b17c0c5470fc06d1bac3f5c48753e7063d93d91f6fe3fe650a53ec6fda4bfd8ff50bf3dbabfbe22e8933efa1fbabf4aefe7bfe5b1c8bf8381c4bfc1b80dbf8cf9df3eac13d83f316503c0961a91bc4c58903f640f813f64a7063f0374d6bf3434ce3fcecdbd3db7e5d33ef8c66c3ffebdbf3f088c80bf70e0873f80a45fbe71c7a13f30f577be1332e53eba3935bfd5f18e3f76962d3c3ba5023fb44494bf44f58dbdc7adbdbf24ad85bf07f999bdfff88fbeeacfca3eb0c2063f2e0f9b3e5e9f1dbf6fcae63f367c4a3f57b271bf70999abfdfc103bf3d9e2ebf7c4f1bbf5bbcaf3fe430c73fa50f1e3e74deddbf213814bfe45b903fd10fa13e0b129cbeafe50a403c09bf3fec1ae9bf50349bbff64f133f4c46293f58d1713fe96bd13eba72703f98ec573fff82663ef52a92bf61ed15be77ef5c3fac698b3e36ff1bbe6b28c03f777baf3e2c5f043d204572bea7972cbfdea064bf9a3159be8b5d38bc13a0883fc25ebc3eacc59e3e15d9e93e77a823becb086d3f94484f3eba4f053f685c823fd1d0023f5ed4dd3e707be33fd1c7293f2a9ae8bfb433083fb7ea7a3e00b1f63e6d5eb0be39e98ebe34c4cabeed59b93ff4f48fbf168cdebf8b7429be14e94c3f4c258e3f333440bfe832fd3f355fbcbf0cc29a3e0e844fbf7b9b923f4cafb73e21aed3bf423403bf81cda63de61a4c3f9d36a63fa62c5abf353ce6bf36e762bef51ca03fc65120bf54e316bf23eadcbf0a4f14bdcd2d9abe98e2823f0dca3cbf9ece6ebf4c1a47c03f107bbf8791913f85e3eabfdd2d2cc00abf813f151c71bf191e65bf14f94bbf9079d7bfb6313cbf2b7b8e3f95d0873d63692fbf2471683f5e9d4f3e4f1f5fbde31d4ebf67a293bb7bb1433f462e74beeb96243fb0549abf8ca450bfd13a03bffc76de3e22aae8bfcb34ab3e8ebbfa3e6170a5bf2d1c003f9293c43d73e08e3f38264f3f8520e43f45c1e03f4c7be73fd4e729bfc0de0d3fd3e57e3e7c21d6bfca1b28bfe700303f0804c73fe8bc223f0226c03eb840db3f7cf29c3e33319ebf99084e3f086d1fbf8b8eacbeabae9e3efc9fddbe1766e9be90950a3e96d4cbbedaa3e33fb03ec1be707ea53dabdedebeeeb9273ef73835be93212fbfe4cc59bfef1baabeb99ffcbf81eee93e20e61d3b0d144bbf9e6e913cee75123f193f13bf3a25173f42648bbef6a6363f7957bfbde9e6bebe7ceec83e9a86ba3fc7fa1fc06e30a1bf8f9c033f136253bfbc64253e82ecf43ebcae02bf8c14f5be8c69bb3ecee4543f25212b3fec3db53e330fa53f7d7194bfda966abfd1cc92bfef6c063e2636ea3e00a2b43f01e97abf712eadbffa4911bf2eb48b3f4e312d3f805b87bf142f7d3fa01630be7c859bbe568b35bf018df2bfcb0b10bff4b05b3f6f69b5bf3941f43e53b1d5bf8ff5003ff216c7bedbf1613f393401c00d501b409a82e03d5bd77b3f3541643f73bc18bc0b27bb3f638bed3e1ac5023eb739c9bebeeb5dbf982451be885a323d4d5200bfbda1b43fd39b4c3ef0a4a63eb48c60bfd6537cbf5971833e0368f33fb00575bca2568f3f9d66e83db88995bef407eabe67c4a13f421c4d3f89e9a2bf4e61723f8a51ad3fdab588bf202707c0ba44553fa57ace3ff6929fbeaa6a113f51208c3f3000b83eb7b50040d988313eeac599bf2593ab3db2e3a03f3dea83bf13890cbda3cd573f05387e3f2dab8ebfbab3523fa789943fb1c9423ff8b0fabe0672c2bf50990ebfeff7b33f6ebb58bf5e1d2d3f4bc3dfbe4db2b43f0ce87b3faff31a3f20a3babf813d00bf1a1b97bf550dd6bfc4b5ec3f135c8c3fd00b59bf01050bbf7fb63bbf6068e4bead7028bfbd5320bf6c17b73e45ed2bbfb7e542bf738eaabf3eb168bf1f7aca3e02c9ef3dcc048dbd19a2563fe7632cbece4babbf2674c73fcf2d7c3f21148e3e95488abef25e66bf4fc8b4bf5a377a3f2350e03f91b896bebe5ac63fbe917abf0e62a33f9d9246bf00e78e3e4b43b3beaf4281bb8617993fbf40ccbf2023353fda6a8d3f13a0dc3fec76103f2b603b3f3992af3f288ef93e5850cc3f92d575be2bc81abd779c373f0cdf89bf939da73f18adbf3f899e6b3e5f322ebf7beacf3e96069e3ead60c8bd5d852140360e573f9b6c43bdd77c623f6b735fbf5d5cc7bfa61480bf3307243f5a81e8be3ba51ec00706dd3e1e401740fb829dbf2e9084bd6136313f3d8cc4be6a6528bfe18e7cbe1bff843f215d2ebf5cd3493f49c4d7beae080fbf221440bf527b103ee327d5bf1dfa8cbf8e782cbd610f074089e365bf13338cbf5eb022bf599b253fe63d1dbf43569cbebc1b153fc61cc4bf3842d3be751b2b3ffefe49bf17b9c6bf5fea203f6746c13e3cc5b93e92dceb3ed9274dbfc9af6abe9a570dbfe293e43dbcd57abf39e10e3fdaa77d3edf60a03fe195af3ce3bb88bf63c60e40b54cc2bfb72ca83f78a8353e308addbf0d1d2e3f24af02bfa8b2a53f7bb041bfa1965e3f7e75c1bf3cbc16c0799480bf5d0b7abf367e8ebfc89523bfbc9c0abfeb72ddbeb0a0d13f619c2f3ffd5221c0b2f7d9be0115013fbb02a7bd8a8f393f54e08d3e44b09f3fdb26cbbf4d749c3e312037bf499a48bf9e73fe3ea62cc93d1fcfb1bf9eb6c23d7e11e5bf59cf033fb8d262c061f6b5bc01783d3f995908c0382a04bff664053fea1262bf487e183ffde1ed3fe9654ebfaf0a8fbf88a586be2c55b43e0cdf38bf41ceb83d6c7756bee7f402c07bf3ad3f2319eebd13f8333e431db0bfec550c3d7da4b43fdbf40fbf6d7ced3e43680f400c079a3fa525f63fa21bdf3ec30d48bfc3ea4fbfb1b2fdbec48ce4be750ea43f06fa223e4d6f07c0cec56abf8d970fbe52d7b4bf807b1b409dae1ebd23befabfd07c8f3ed05fafbfbccf243e03f209bf602743bad9c4ca3d04b334beb412303f5df90abf93c4e03e27ae23c0a2d648bfb6be743d79364c3f32a609407d52fc3e434aa9bf40c0563daabf04bf7c4d92bf2c64893f4d714a3f37033b3f5a1b31bf15bf5f3f2d46a5beac5950bf3a00e83e8f24913d49ff9b3d67fb68bed6d8b5be18cdb43e1b3c463f140bc5beafe7fa3e35e0eabd06e6533f43f34d3ea026c93f8262fb3d96b817c02eef843f2d010f3f52c5523efeb69abe1cfcb7bf5c52f5be9e9a4fc0b9ddc73fc78616bf5614f8bf6d60813fbd78093fe128d1bc2d2b78bf5a4b47beb3372e3e66725b3fca6aa7bb4c6805bdf17ce23f68ad1ebd4e3505bfacfadfbd8612403f6e6e8bbfdff8b63fe290aabed00c573e682f9b3f7b1c39be1400babdf6921abfc7496abf1bbd923e87869f3f46a38c3e1cd616bfb61eb3bf2e44153f999ae83e0ee0693f84a3e03d73e02a3f430a033f04eb98bf21d387bf680732bf21117e3fd220bebfe5588dbf10f2b0be1316e3beaacf0dbf7a0baf3e418ae33e5b57323f3b17ecbe93ac20bfa582e93f037e3c3f16dfd03d6ce6fe3ef73b053f762a5e3f90c125bd1b161fbf2db3f1bd6e0f12c055fe84be51e8183f413826bfb18fa23d24c5393f7527393f64019cbf0d587abf07c4a63f3e97b33ffc778f3f09f2c83e871843bf81fb9dbd4939933ff74dcabea01c67bf37d70bbfff42e23fcb64bdbe6b7725c0de8c04bf6c8e8f3f7fb0c23ffb75064076a8ea3ed9dd3ebe3f6becbed325913f4d2068bf75eb3f3ee89c61be7963fcbfdaadebbe383b1c3f3b0189bf3629803ef3bf53beb66b54bfa6169bbfdbf4aabfa08b9c3fd053a33fbf9555be0e7320be8b4ecb3f7cf4a23fb649f8bf33ec7d3f7eee0540df8f1f40795ab43f4cc147be52a2f1beb182b4bfb3f0173f85653c40b485d03f624b7fbd9a95713f34a140bffde40440869252bd55ba00bf5855e3beb0818a3fc839033f04ab593e894b6bbf6123c5bfedf11a3e5bb1afbefd7093bf017c3cbf48b3acbe358a8abfa6299ebd991b6abbe4f9113f519ed5bf23ae8e3fcdc215bf96ed6cbd79908a3ec5328e3fb87397bed645823f981f20bfe79677bf8c5df73e7a1dffbfdc37b9bfea6da93e8fabccbf596042bd6ff6b3bfd0c0a3bed83073bf519c5cbd248515c0222bc03fcfa02f40ad1866bee73ad0bf417ed6bff3ace03e6793b0be570dbd3f074162bfae28e13fdf5c1d3f43c66fbe06f308becea969bf94ae1e3f5b48e13ec2b30dbf7a16ca3f58f9163fe54c063ee3321ac02b6e4cbe9118d93da05dd4bddfd2f93e03932e3f0b5184bfe1775ebf37bba93f1e9202bec1fc903f6ecbc43e320e773ec58301c0f920933ebbdad7beff83a7be253692bf69d089bfa84aedbe0ee8213f26ac51bf63d6d6bee2bc853ebfb9933f91a9c6be127b073faa0769bfb8d208bf8965393f26aff43f887b35be0a26fa3f7dafee3f2ee5823f6321aabf0be50ebf5c3fc2bfa2cb723fa6cdf13e9d4a74bf8e0c003e310da53e61be933fe5eff4bee44cc0bf1f6a8b3f1ca7f7bf082c82bc8bcdbf3fff1d9bbb03a3e23d334284be2cc7e33fcbdea7bfc36e573f03d9f2befe95bfbdec562abf800c823f3f7f613fefd28cbebc9e99be6badc1bf3425ee3ee91392bf11c5903f2256b73f412ab93dc4283a3f8bc2b83f519132c0954497bfa3c93fbef3467abe50ef5a3de97d25bfa6d2df3f936172be39080d402ca882bfb4fe82bf88389d3e06dac4bd39841e3f79a5d23d0e0f953f4f06b8bf44d4b43f5f8c08bfca9a06bf917c963ff90e3c3e8af0bd3fb7b5a7be806162be90d3c13e30c9bd3da01201c0fcbe0cbe57e8c13fac5f7d3f784acabfc59498bdc5f80a3f196d0e3fb9a76c3fc728c1bf9bce25bfedb70fbf5070a4bf16ca113e3236403f77a710bf06ef0840b8d2a23e0b1e203f8d9c1c3f77c1d8be0adf4bbecd02c8bf31a22dbf50dc6c3ccdba22bd959fd53fa002823f358dc5bfd077893f45e6d13fa46f923f49749abf6e00b63e6c20a1bff14491bf623023bf22d0ee3e8162a83e05333dbf9cd8283e04155bbffae8cd3fafe8fdbe8bc9b7bf2a4161bf8c5fb0bc4c8494bfa4727dbebfc3803f748c47bf0ed2f43e3f648abfa64ec6bfb9ef9fbfe46718c0daeed1bfc22c0640c7594c3efc1308bd75d1083fbe7ea8bb5b8ba0bf9b25d53f8995e6bff559123f74e1473e0157213f0cc803bedb31733f2bc94d3e30ba0e3f15b77abfbbf1a73ede308b3e6341e5beb7693ebe2ee624c0c144cebfd44f8cbf5d1d953f406affbf9bb63ebe3c070d404a4bf1bdd3bf74bfb17d9e3f6a8fab3fde46d0beb3e997bea388eebfb75f5b3e007cb23f3633b9ba3e21023dc5f8173f7007793d2f5661bf49e8b9bfa293853fdd7bbb3edb93743f8ff5553df3393a3d61c3a63ec7548b3eb84fa53feece5bbf62418fbe6a13b0bf09a610bf0a11313fa95a913e43a146bf5346ccbbfb12fbbf64dab43e433af6bef0c1ee3f99bb6dbf86f01fbf1bd068be40e08d3df94373bfb6e84abfacd95b3f46d89f3e82c687bf64a18cbf6cc69b3efb453b3f731a653da9349dbf2d144440e0d1a9bdc6691dbfbda2563f7f197fbfe05b6e3f5b7c8d3f3895273ec3673dbfeee0d3bf05c80940ab1af0bfb586eebe9d194a3fced9123e1f3fd4be5f3a203f54e39fbf5d26cf3fd374433f70c25c3fca6e36bfbc1de2bf785611c0f294e23c971ea23fd3415abf0c1e3e3f9693843fb36e0f3fae76d5bef304ca3e555dacbfd05a22bfc1dc9d3f20fe52be643a85bf0734983dabf914401fc6863f24d2ad3eabc5c6be4e8a163fb257eb3e75c3d03fbae821bb2f1ed3bdcee6a73e783d013e10bca53fe31827408919a63f8f2e03401289d2beec9dac3f5742f03f4d8f8dbe22bb443f24fe283fb941a23ff8e8b43f257c263f9e3a14bde29cb43e35d9613e08ae183f3d390b402508b23fe1cbf5bee96c073fee220c3f307696bf0325563e2cbe26bf5a5396be9c1599bf341e923e3acb2140813f703fb6ea993ee81918bfa3c5cd3d4abea3be0e4b3e3fdc82123f0f5789bfa8afc93e92b4cfbeb1dd6bbf3f95723fc7d30abfc4b058bf738b86bf6ff4a0bf607b97bf938d2b3fce66053fc5a2b23f6448bcbf1ec6c1be9d7315c0d665cfbf4c9bd93fef0743bf6fbf8dbf73a5583f17b6743f8fcc623f390f7d3f81d93a3ff528b93e8086803ffcd2343f6edb1e3f6d469dbec08ba9bf078e19c0745393be5b7e353e5b39cfbcfed000c0042b9f3f8cc58fbe250f213f93c696bfc177b3bf0b60543e7048d03e7753b63d3c931ac0f8f6643fd30885be954912c036dc8f3f28b0cbbfd1e4973ffc929abda31a42be1b120cbf3db06d3f6bfa02be8a7d77bff8b13b3e5e8bfb3e45bf8abef69f323e8acbaf3f7383c0bb3f60e1bfeb6a083e17c1bdbf369047bfee0689bfe827273d477f4ebed4c9f1bf9091ce3b0b733cbe0bbe763fe9f33e3feca205bf64b676bf9397d23eaf23523d94dcf43f90381f3f4601963ea7792dbf3933853fbc2a363f8030953e1c6bb1bed8530abd2ac9bf3fb17f9fbfc3a54dbfa982243fcc2e483f2aa24bbeafa495be6e5b2b3dc33bf63f6f2017bf65b8bc3f60e8c93e445d503ca949583f871dba3f3e68de3d49b3083f2ea4283fe63bc1be838cd7be260b393dcab2adbfcf4d993e1c9f3bbd6e48a53fe82cc6bebf6ba93f944293bf36b2a9bdd2d5423f51af8fbf8072e0bf5d7920be4d900cbf7c5b88bfda9e0f3e24d14cbed4d6843eaddd033f929b9dbefc5eebbd7b0f973f95c510bf6ec126bf82b005c04a961740bb70c5bdf7cefd3f1f6f433f5483123f9f4a97bf4690f2bf3532a7bf315759bfafcbbcbf5f30a3bf06cd423dd8bd65bee4258bbe38e28dbd3d916bbfab2ea6bf1b0d45bfab279dbffbe172be591896bf53f638bf15825cbffcdc8ebf0850a4bec7a7d03d587bd7bc0c60acbfd8ee74bf29632e3f7d7a3ebfd3d5583f105b003f86e4d83c7b47dc3eec8bc63ecaf5f7be662b4b3f2b66c73fc855b63e9e5c303f9e83babf87753a3fafd46e3f843d31bf4b968dbd5fc4ffbe83e4aebe8c2ba23ea1777bbfaf9f15bf084724ba3e94773f1d2712c0eb5d043f56d294be50a2d73c985198bf3e30dfbebb71babfd5e1873fe2eed7be791122bfc1f9ac3ea9e0d3bfe69339bfd795243df0674bbff2bdd13d5bbc14c06b40b3bead21c7be53dad8bec49389be633acdbf70e3ce3e1436febed46e33be5632b5bd5dccdebf5497c1bfef995ebfc401f33e7ea949bfcb79b2be31c34b3fd382a33e514b09c096f3893d19b1f03f2ecc07be73cc7cbfed3e48bfaddfc73f22c54e3fe3530a3eb1c8d63e30fa5b3ff9bd8dbff404823e07d813bfb8fbe03fce7fd93fc6c45a3f0320493f82b8b4bd8511bbbfcf1fbd3e1d743cbe51f901be48c154bf82e8be3ec600a4bfdf3ee73f9640113f4009d33eb4730140f454c83dc34733bf5861143f7da3e8bec03ae23fc699b1bdc3ae08bfc290963e33610d3ec2d9cebf67e2223f5c27983cd03086bf7989213fc535c5bc0b6983bfb6ef04c0774ed43cfd290c3facdeb6bf7362093f2f677cbf22ee0d3f2e47083e531af4bf53801440af0281bf8d3cfd3e7402ae3eda82fbbf34749fbf54ce6d3f2d52bcbf8ab992bd37adcb3daebb2e3e099ad2bf38380bbf61615a3fa918cd3fc4bd93bc93c99fbff290a1bfe472a13cc2a4f6be71536e3f014e8abf3d136ebeeb9469be6708813fb106073f42610e3eeb760ebe97d08bbe090b103eb08e903fd5de763eba91b33e5af5adbf2db6cfbe91e51f3f8304b63f6d2735bf5b090dbf05220c3f4b9892bf6c15cb3c99ab9ebfa90b263f3a1a75bf5549e63fc3fbc3be8be2b63f0e81b7bed024353ff9f19d3fa0cbdc3f1db6fc3fe280363edc123cbe6acf8abf25edabbf3182073fe2a684bee407be3f669ffcbd2f0811bf64fa493fd10da2bfbe1126bf6c032a3fe2a0cdbfe751e23fe08e883ed3a9b8bf4dcb10c0917eb33f534310405816323fb16f8f3f21bb49bebfd39d3ff5f0a53f7ca8a73d6bffc5bec52c983fdf250bbfca3c54bf912506bf72fa453e1290673e2cbb89be4032cb3e8af8263e204c6fbf16ca833ff5dd363f8ec3ccbeee6a11bf37162f3f4c580e3f9f513d3f9c520cbfd8dfe3bd924e233f18ef353f450aa1bf32ea1d3fcc1b443ffabbaa3f6250ae3fe7891dc08e3b8b3e527f053f261d513d0fa848be4a7424bf260ee1bed769d53ff02ed63e53737b3fbabe8c3f04ac1b4045ab2c4036abe03de13505befa3b73bef413d0bff09b7d3fffabb33eda056a3f0dabc2be6c439dbffc468b3e51e28fbd63188e3fa250fcbfd8070fbf6569d93f4b938abd5b3423c0e39fdf3eaaeff13e19ca08bfb43d9e3edf52be3f5e6f963f822f91bfe65f17bff982083d10eab53c7b04c8bed960d7be725c3ebf161f1c3f641d0abf10d33e3d7a93303f3564c13e55b761bfc9a0cebfe0c3b7be146be93eb84ef5bee4513d3fbfe1293d2d4277bf1c42ba3e779c9a3f24b9313f9e7410bf6ed791bf860dd2bed32dad3e3d0d0d3f7f952cbe271b31bf73f4cfbe7d8a71becc35ac3e9eb2073f422dda3e0ead01be2d2dabbe2211b9be5fb8843f513d33bfca7d063f64cbdf3ecb317dbf3f4128bf0c39863eb5bc12bf69ee243f19ba013ed1a2d1be646cc93dc0f082bf2db6403e035ea73ceff9dfbec8f6993f83bc293f310b82bd9da4963f75dc753e6766863f70c25ebfbda21abf146fb93d19267a3f87ceffbe085395bdc92f3abe5027adbfd5cd8bbd86084dbfbf5a153f378120be83bd763f01787440b662eb3ec0cdf5bfb40f63befd99563fef0d3dbeabcfe83fdc62b03d26688cbf339400c018cc21bfaa2a953ee09390bf9261ffbf785eacbde4b81f3ee35dadbe824fde3f5ea49bbfc353aa3fc08fa6bea1f791be9b074f3f5541a63fbf7efbbf3b269dbf14aa623ff3c5f1be655d58bfae719a3f82fcae3e878456bfc0629f3f04038bbf67997a3e6f470740de2e7bbffafa5a3db262893e5f23033fd0f77abf0b0d01c04862f7bf92182ebf9bb7e9bf05df8c3f84400c3f296d0a3f07d1b7bfdddeee3f45be6b3defacdfbda033bc3d84a589bff0b386be8bdf7cbd810c0fbf80c782be7278c6bf5450a53f3e1c3bc0a5b17dbf648fa8be17f3a13f2bb18d3fa0575c3fef9d383fc67491bfd57dea3fdf0a523f62caff3dbe4619bfd47dc4bec818a3bfe081a5bff1bb8e3f7036bebfe906ae3d484293bf2087c93efca54cbef2f57bbef2a49abfeebc783fc9c0aa3f8e97573f032d21bfdacff63fc2855bbef61a933ec1f0b1bfa14f9ebd4799af3f5b1f8dbf195421bf8706233fd98f8f3f247a1e4053d68b3f7e248fbd8402943f8f1f9c3ee5d8a43f98e95ebf800a283f6b5111bf948377bf509793beba58a0bfd08e823ddd460340375e7a3ea4f8ba3f06871abf15eb10c0fc16613fc125b0bfde4d9cbddecd893f327aff3f61fcb8bedffc393f4927bb3e50515ebfac4617c0c015013f6b8a823f891653be9939843ee1bfdebdfb09264059bdc73e15d122bf9f849d3fdc3591ba3d0fb3bf50e1903f8611f2bfd49fd33fea5274bfb55222bf428f77bdae2a10be08059b3fc8ecde3e60b8f03efb3f3bbf8f5945bfc0eff33eb1ae7cbf5d2efabfc3dad4bf1daaabbd7124ff3d22a30ebf766d90bf0ab3823fc38616bebfaf90bf9ecec8beb4c8d3bd89f8273fe83440be84e1b73f8317493e11eb2dc04dfe023f681dc9bfffc9db3e2ebb073fbef5edbf55e046bf11f1913eac84d53efa38963fe9799fbfee1759bf1e85bebc9330a3bf0fbbac3f0df7d6be695b1e3f908e883e278b71becaf78dbeef9007c0a8dba83daf09533fbfc5d7bf025ba13f5cf0adbeb13f03c0693736bfa01d1dbfed5636bfc76bd43d395fbc3fc6c3b4bfe036f1bff3fb3cbff7885d3e337bafbe28035b3c1a17b2bffda4ea3ed8c4323f9942433fef48233f2c560b400599953fb27587bfd10c933f8ab4a23ff2c4c83e144964bfb18c0bbe38f05abfe60db33ee1d1e33f683391bfaeedb43f7ba92ebf34aa8cbfd6dd0ec0aadc903ff0ec90bec8cdd13ec0f60ebf687a3a3e7620e7bf50a4fcbe089386beec595a3f6a9298bf35eee53ea59e99bf2857903e9583063fecbe223f45623bbf1bd5093f016e4f3fe42caebe280dc63d21ca1e3f8e29533fea05963fc61a493e5aeb593f7f56b9bd08ea08bf2a980640311566bc650d4e3fe319a03efc3cebbe96b9b63f38bd5d3f405693bf235a8e3f25dde33f4e9b25beb3fb94bf2af1cd3fdc9c66be45b854bf4204dc3f4ce6c53d2caf92bf02c9d63fa1622540e13757bf0355c63e57c4e1bf79c9d5bf6a93e53f6807183f7c640a3f1a47a7bf7a87d23d5b075fbfc9e78fbe2993343f214afc3e2da5713f90478f3f00dd6bbe30406fbf9f13813e15010e3d52d701bf65d652be07c08b3fa9ae663e0761833e044db9bf94a366be331dd9bd3c50b03f34852dbfb20ca1bfb73bb6bf97004a3f7088cc3fc8a515be343c40bef1d6973dc87459bf476b423fdd25983e1cf6283fa11ef6bf606cbcbf3a1cdebe1cce3c3faaae123f46a7c1be7e233abee1b082bf4ca6d43d7ac6993ee48ae33f4d89b4bf99b0143faf4b89bf416a8f3f2a0a923fec0abebf2770aebe9e7953bf5ab94abf571fc6bffc5accbd740e623f4c49c63ebc992d40e25b7f3cf3fbfdbe50e65c3e7438093f05b21abfe2a1803f0ab6803e253289bf322b963ffa0350bf8a1d4b3f313b1cbf74b46c3b464fb7bf43dd12bfbb3b99bf3ec56ebfb2370c4045ba86bf9bec53beb3a38fbf50df1bc00177563fa011373fa3e9063f69f0eb3ebe38b53f0a4fabbf68570ac0b470ca3f4756543ef1634ebe066a793fc652a93f7b03b03ffbc5293f23c59a3e1f38eabfbf31c43e3bd5cb3e6cfcd43e4a83a4bdb8628fbff8240fbe5a83c5be9e1e67bf7825d4bff2ee603f9b44243f37e6093f61086b3f48f7553fd5fbccbff08bc6be2140c8be5891193f68746a3c8f36b03fc528cabdced3b43e4f7583bdc67f35bf7899223eb1d498be8739b13f2f6b263f95daa43efa4bfdbd26bfd13e80979b3f455c3cbf11a2d4bcb347a83f13dc75bf5a44623f6741833e158909bf46c1c0bcd85be5bf97dc04bfc4ed393fc81cd3bdc10939beb6cb11bf40cbd9bf8946e53f5f22a7be257fb8bf06328b3f80a09dbd4a836bbfe2b6123f063ab1bfd612d1bf02c5a93fd89a233eaf164cbf6724fd3eeb615bbf1b9e7f3f82f6fcbe8ddc0bbf30cfabbf9faa373fa73dd2bfb1afd4bea1dbb3bf95134c3f917fbd3e5d7405bfde748c3f8a4105bf31713cbf807749bf7376aabeee1560be8764113f5fb9bc3f1c6fa9bfbd5460bcb178d23e09f5d7bea588dc3fa66dfabe029ae83f794cee3ef05acdbf41c4d3be8c0412c098c3b93f25ce23bed42c1e3f454b073dc3b30140dbecc23e8cb8e6bd662d0b40eef376bf5d860440f9f2063f72ebd0bf63dd98bf1cfb8dbe4fff21be8170513f2d198dbea1a3283fb1776abf75915c3e5f661cbfb52456bfcff3c83f199f3cc09f1cecbf94deb83fd24edfbf4d58603fe77562bfa8b13cbf134e343f6368e13e7495ae3c0bbbd7be57faa9be7db6ab3ea40e003f9098933fe4a3dcbf3aaba53ebcf84a40ab8398bf34f7b0bccc5309c04f9439bf8b0dc7bfff25763f63f0adbe537963bf12618e3f8be474bf668d863f8e5f7cbffaed003d042fd83e538ee63ef253c5beafafa5bd781816bf536081bf03ac6c3fff132ebf56cb46bf290761bfd12cef3fd8268abf758287bffe5d6fbefc38a7bfe7d2963fc7750340589cf03f9ec685be3ae472bff9e6873f14c5993f98a213bfdf652cbe28f0e5bec23e69bec5f07e3e8e8dafbe0da7f73e290c3440dce712bfb7f613bf959b7abec72b183ffb3ea93f7cd78abfcc10a53e937210bf89f930bfeb1868bf7035e33ed5d0a13e740e71bfec43733ea56f32c0e0bfe8bd55171bc0fea8f33f9637de3fb0eae93e8ce9d73e059000be07e15e3f01fdac3ee3ef413f1fbdcabf9322ba3f3f2337bfedc57ebe65a3b83d2f3e433fecff38bf086f0ebe1c4e223f47e04fbfdb9f14bfe02e7b3ede25063eab00ec3f3982fd3dde021dc0e41674be5793133e62358c3f7057b43e38db7b3ff18bc53e661faa3e627ac33e10e074bf74ca493e94c278be46468b3e0256373f39b42c3fe5bd78bf282432be4347603fc8bdf9be597bf7bde7a08fbfbe3ea33e832d47be9099d7bffa3e993f3c12a6bfc732cfbe4022fabee7e420bed16587bf8845713c695828bf86cc31c08a4f1bbd60d3ef3ef5e390bf6c40c2bfb5e3e63f9b7dabbf9515f43e00c5b63df460233e7f636c3fa02d883ec32c173faf93afbf42b288bfb22a78bf6399044037545a3e997d4dbd520efa3e5d99dbbc59b495bf5ff5a2bf001b92be38b4d1bfd39c3bbf102e31bf74f0553f5016cb3f4ee629bf0b66203f41f544bfbd1f8bbf6a76a1bece4bafbf37399e3e5f186a3f9a7806bf5f4cf3be43df6b3f576ddc3d7ec4b1bf503eeb3f96a237bed12c383fba66f0bfa5151fbea3f6b9bf7459903eba2414bffbf23dbf3b1e31bf4a87cebe30b659bf0ae61fc02a6fbc3e78b28b3d5b34f3bf9bfb2abf8c19d83eff76b4bf52ca883e2e5f623fede9603f1c9e013f046f29c051831ebf2e5920c0587c463f5266d63f400e32bf148843bfc7f51d3fda2a67bf2d2de0bd0b6be9be70f81f3f466037bf860dbcbf13b0e23e5b448fbf66e5a5bcbc77be3fc03893bff6161b4068d89cbe1260f4bf5130af3f2d47a93ffe5504be2e8cbe3e225c2fbd5a504a3fde0ca7bf2863ad3fd8b0a13d1c8deb3ee7718f3f5b535bbf0ea1263fe7e3bbbfe971603f5772a53eeb8947bf1d990fbf0f5df83f802de5be01f116bed020873fe47c83be89ffc5bebd307cbfd7b5b33fe50c08c00bcfc83feb7593bfd66206bfadd189bf1aeb1abfbb9edc3f3629fcbea3b09abfb35114c07b6dd43fb57e103f2f3aba3f008abf3fbfca3e3fab900c3eddc477bf9e2e003fa6f8f2be76584dbfc30caf3eaf734a3d24852440af5b6c3fa2dfe0bf4763043f7c4e1740e4d183bd98b5893d9fb704bf2c19043f81e7afbff89ade3fff1b213effb0413e646c45bf87daeabf443e3cbfaf0be0bd42dbe83f9f0b6a3fe5f03abfd6c3873fad6bb5bfc00c6dbf1cdfb5bda25e9a3dc5c4453e1d07523f64c486bec260823cb7c319bf56e2f6bee04b41bf233b3a3f76438a3faae2be3e4cd44cbfa8007b3f19ab2a3fc34cec3e581bee3e6a44c3bf7feba1be25a13dbff32a3d3e3c31893ee246ea3e0867153fe922a9bee0a621beecf4c63fdee25bbfd8a58cbed010d2bec2b6d43d9c3caa3f9a2679bf2d0675bed012bebf082015bf3236a43ef093a5bf7cfd0cbf9d1366bf1acb22bfe1e16fbf13b83b3f8d963c3ecc926ebe79e0354045390b40cb16a53fdab0933fd8b2dc3ed8ed063e289a56be3dbed53e2c7a353e2d2d433fb35df53ef0f1b6be2e149abffc1a013d6b26b2be0a69983fec4d853e6da47c3e6a24923eacd3873f9de920bfd93effbec2e17cbf0dc8fabf189104bf89b823bffc39f93d505e883f1789e93e4903953fd87d31be26fd193f35be104061d0463fe840683fd4197abf70205a3f816c06c024fd943e0381e4be815bef3d05438c3ed832fb3fc715a83f6415a3c059d5bebe19a5d43e8a03c3be0741ca3fd78deabee900893ec6453bbf271d5f3e2c8206be5dbfb5bdf9e30040fe5116bea587a2bf32eaef3e47c73cbf9573243fb8fa74bebd596cbf7643093fb2d0a8bfe8980ec0f34ac23f863b8c3ee25f783ff6e34140cb931d3ec0c51eba9847053f9f3543bf64ea2240e99790bf2b2c50bfac43ec3f2758913f7fe493bfc7de6f3f4973473e9aa8613fff76cd3ff71302404ab58d3f"), dtype=np.float32).reshape([1, 3, 32, 32])


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
